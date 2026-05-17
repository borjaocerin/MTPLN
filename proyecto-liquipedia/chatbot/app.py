import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging


DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Keep terminal chat output clean (Tú/Bot) by silencing noisy Transformers warnings.
warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`.*`max_length`.*", module=r"transformers.*")
hf_logging.set_verbosity_error()

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from chatbot.ingest import (
        DataIngestionPipeline,
        DEFAULT_DATA_FILE,
        DEFAULT_STORE_FILE,
        DEFAULT_TEAM_URLS,
        SimplePersistentVectorStore,
    )
else:
    from .ingest import (
        DataIngestionPipeline,
        DEFAULT_DATA_FILE,
        DEFAULT_STORE_FILE,
        DEFAULT_TEAM_URLS,
        SimplePersistentVectorStore,
    )


class EsportsChatbot:
    def __init__(self, model_name: str | None = None, store_path: str | Path | None = None):
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL)
        self.max_new_tokens = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "96"))
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_FILE
        self.data_file = DEFAULT_DATA_FILE
        self.vector_store = SimplePersistentVectorStore.load(self.store_path)
        self.generator = self._load_generator()
        self.records = self._load_records()

    def _load_generator(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            except Exception:
                # Fallback path for models that fail with trust_remote_code/custom kwargs.
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("trust_remote_code", None)
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **fallback_kwargs)

            model.config.use_cache = False
            model.generation_config.max_new_tokens = self.max_new_tokens
            model.generation_config.max_length = None
            if model.generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
                model.generation_config.pad_token_id = tokenizer.eos_token_id

            if torch.cuda.is_available():
                return pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )

            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar el modelo de Hugging Face '{self.model_name}': {exc}"
            ) from exc

    def _load_records(self) -> list[dict]:
        if not self.data_file.exists():
            return []

        try:
            with self.data_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []

        if isinstance(payload, dict):
            records = payload.get("documents", []) or payload.get("items", [])
        else:
            records = payload

        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    def _find_team_record(self, question: str) -> dict | None:
        normalized_question = self._normalize_text(question)
        if not normalized_question:
            return None

        candidates: list[tuple[int, dict]] = []
        for record in self.records:
            name = str(record.get("name") or "")
            normalized_name = self._normalize_text(name)
            if not normalized_name:
                continue

            if normalized_name in normalized_question:
                candidates.append((len(normalized_name), record))
                continue

            name_tokens = normalized_name.split()
            if all(token in normalized_question for token in name_tokens):
                candidates.append((len(normalized_name), record))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _parse_requested_count(self, question: str) -> int:
        matches = re.findall(r"\b(\d+)\b", question)
        if not matches:
            return 1
        for match in matches:
            num = int(match)
            if 1 <= num <= 100:
                return num
        return 1

    def _extract_year_from_question(self, question: str) -> int | None:
        match = re.search(r"\b(20\d{2}|19\d{2})\b", question)
        if match:
            return int(match.group(1))
        return None

    def _is_movement_year_question(self, question: str) -> bool:
        return self._is_movement_question(question) and self._extract_year_from_question(question) is not None

    def _clean_movement_text(self, text: str) -> str:
        text = re.sub(r"\s*\[\s*\d+\s*\]", "", text)
        text = re.sub(r'[\"\'“”‘’]', "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _format_movement_response_line(self, movement_text: str, team_name: str) -> str:
        date = self._extract_date_from_text(movement_text)
        if date:
            description = self._clean_movement_text(movement_text)
            return f"el {description}"
        else:
            cleaned = self._clean_movement_text(movement_text)
            return f"{team_name} {cleaned}"

    def _filter_movements_by_year(self, movements: list[str], year: int) -> list[str]:
        year_str = str(year)
        filtered = [m for m in movements if year_str in m]
        return filtered


    def _is_movement_question(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "fichaje",
            "fichajes",
            "movimiento",
            "movimientos",
            "transferencia",
            "fichar",
            "venta",
            "comprar",
            "bench",
            "benchado",
            "roster",
            "plantilla",
        ]
        return any(keyword in q for keyword in keywords)

    def _is_upcoming_tournament_question(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "torneo",
            "torneos",
            "partido",
            "partidos",
            "próximo",
            "proximo",
            "siguiente",
            "calendario",
        ]
        return any(keyword in q for keyword in keywords)

    def _is_participants_question(self, question: str) -> bool:
        q = question.lower()
        keywords = ["participantes", "participan", "equipos", "equipo", "plantel"]
        tournament_words = ["torneo", "torneos", "major", "iem", "pgl", "blast", "esl"]
        return any(k in q for k in keywords) and any(t in q for t in tournament_words)

    def _extract_date_from_text(self, text: str) -> str | None:
        patterns = [
            r"\b\d{1,2}\s+de\s+[a-záéíóúñü]+\s+20\d{2}\b",
            r"\b[a-záéíóúñü]+\s+\d{1,2},?\s+20\d{2}\b",
            r"\b20\d{2}\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None

    def _extract_tournament_name_from_question(self, question: str) -> str | None:
        normalized_question = self._normalize_text(question)
        if not normalized_question:
            return None

        tournaments = []
        for record in self.records:
            for tournament in record.get("tournaments", []) or []:
                if tournament and isinstance(tournament, str):
                    tournaments.append(tournament)

        seen = set()
        unique_names = []
        for tournament in tournaments:
            if tournament not in seen:
                seen.add(tournament)
                unique_names.append(tournament)

        best_match = None
        best_score = 0
        for tournament in unique_names:
            norm_tournament = self._normalize_text(tournament)
            if not norm_tournament:
                continue
            if norm_tournament in normalized_question:
                score = len(norm_tournament)
                if score > best_score:
                    best_score = score
                    best_match = tournament
        return best_match

    def _unique_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        unique = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _build_movement_response(self, question: str) -> str | None:
        record = self._find_team_record(question)
        if not record:
            return None

        team_name = str(record.get('name') or 'Equipo')
        movements = [
            m for m in (record.get("movements") or [])
                if (
                    isinstance(m, str)
                    and m.strip()
                    and self._is_valid_movement_line(m)
                )
            ]
        if not movements:
            return f"No encontre movimientos registrados para {team_name} en los datos disponibles"

        year = self._extract_year_from_question(question)
        if year:
            filtered = self._filter_movements_by_year(movements, year)
            if not filtered:
                return f"{team_name} no tiene movimientos registrados en el ano {year}"
            count = self._parse_requested_count(question)
            if len(filtered) < count:
                return (
                    f"{team_name} tiene {len(filtered)} fichaje(s) en {year}:\n"
                    + "\n".join(
                        f"- {self._format_movement_response_line(m, team_name)}"
                        for m in filtered
                    )
                )
            selected = filtered[-count:]
        else:
            count = self._parse_requested_count(question)
            selected = movements[-count:]

        selected = self._unique_movements_by_date(selected)
        lines = [self._format_movement_response_line(m, team_name) for m in selected]

        if len(lines) == 1:
            return f"El último movimiento de {team_name} fue {lines[0]}"

        return (
            f"Los últimos {len(lines)} movimientos de {team_name} son:\n"
            + "\n".join(f"- {line}" for line in lines)
        )

    def _is_trajectory_question(self, question: str) -> bool:
        q = question.lower()
        keywords = ["carrera", "trayectoria", "historia", "trayecto", "career"]
        return any(keyword in q for keyword in keywords) and " de " in q

    def _extract_player_name(self, question: str) -> str | None:
        # Busca el nombre que normalmente aparece después de 'de' en la pregunta
        match = re.search(r"\bde\s+([^\?\.!,]+)", question, flags=re.IGNORECASE)
        if not match:
            return None
        name = match.group(1).strip()
        # Recortar palabras de cierre y artículos
        name = re.sub(r"[\.:,\?\!]+$", "", name).strip()
        name = re.sub(r"^(el|la|los|las)\s+", "", name, flags=re.IGNORECASE)
        return name

    def _find_movement_fragments_for_player(self, player_name: str) -> list[dict]:
        if not player_name:
            return []
        p_norm = player_name.lower()
        fragments: list[dict] = []
        seen_texts = set()

        for record in self.records:
            team = str(record.get("name") or "")
            for m in (record.get("movements") or []):
                line = str(m or "").strip()
                
                if not line or not self._is_valid_movement_line(line):
                    continue
                if p_norm not in line.lower():
                    continue
                parts = re.split(r"\s+y\s+|\s+and\s+|;|/|\\n|\\.|\s+e\s+|,\s*|(?<=\d{4}):", line)
                for part in parts:
                    part_text = part.strip()
                    if not part_text:
                        continue
                    if p_norm not in part_text.lower():
                        continue
                    if not self._is_valid_movement_line(part_text):
                        continue

                    cleaned = self._clean_movement_fragment_text(part_text)
                    if not cleaned:
                        continue

                    if team and team.lower() not in cleaned.lower():
                        source_text = f"{team} {cleaned}"
                    else:
                        source_text = cleaned

                    if source_text in seen_texts:
                        continue
                    seen_texts.add(source_text)

                    fragments.append({
                        "team": team,
                        "text": source_text,
                        "url": record.get("url") or "",
                        "raw": line,
                    })

        return fragments

    def _clean_movement_fragment_text(self, text: str) -> str:
        text = re.sub(r"\s*\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _parse_movement_date(self, text: str) -> tuple[int, int, int] | None:
        # Extrae fechas en formato día mes año y devuelve (año, mes, día)
        months = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "setiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
        }
        match = re.search(r"\b(\d{1,2})\s+de\s+([a-záéíóúñü]+)\s+de\s+(20\d{2}|19\d{2})\b", text, flags=re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month = months.get(match.group(2).lower(), 0)
            year = int(match.group(3))
            if 1 <= month <= 12:
                return (year, month, day)
        match = re.search(r"\b([a-záéíóúñü]+)\s+(\d{1,2}),?\s+(20\d{2}|19\d{2})\b", text, flags=re.IGNORECASE)
        if match:
            month = months.get(match.group(1).lower(), 0)
            day = int(match.group(2))
            year = int(match.group(3))
            if 1 <= month <= 12:
                return (year, month, day)
        match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
        if match:
            return (int(match.group(1)), 0, 0)
        return None

    def _build_trajectory_response(self, question: str) -> str | None:
        player = self._extract_player_name(question)
        if not player:
            return None

        fragments = self._find_movement_fragments_for_player(player)
        if not fragments:
            return f"No encontré movimientos para {player} en los datos disponibles."

        # Ordenar cronológicamente según la fecha extraída de cada fragmento, con fallback al orden original
        for index, frag in enumerate(fragments):
            frag_date = self._parse_movement_date(frag["raw"])
            frag["date"] = frag_date or (9999, 12, 31)
            frag["original_index"] = index
            frag["clean_text"] = self._clean_movement_fragment_text(frag["text"])

        fragments.sort(key=lambda frag: (frag["date"], frag["original_index"]))

        # Deduplicar movimientos idénticos
        seen_texts = set()
        unique_fragments = []
        for frag in fragments:
            cleaned = frag["clean_text"]
            if cleaned in seen_texts:
                continue
            seen_texts.add(cleaned)
            unique_fragments.append(frag)

        context_lines = [frag["clean_text"] for frag in unique_fragments]
        context = "\n".join(context_lines)
        prompt = (
            "Eres un asistente en español. A partir de los siguientes movimientos del jugador, "
            f"extrae la trayectoria de {player}.\n\n"
            "Instrucciones:\n"
            "- Devuelve exactamente una línea por cada movimiento válido.\n"
            "- No uses numeración, listas con guiones ni formato JSON.\n"
            "- Conserva solo la información de movimiento sobre el jugador indicado.\n"
            "- Mejora la sintaxis en español.\n\n"
            f"Movimientos:\n{context}\n\nRespuesta:\n"
        )

        try:
            result = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens + 30,
                max_length=None,
                do_sample=False,
                return_full_text=False,
            )

            if isinstance(result, list) and result:
                generated_text = result[0].get("generated_text", "").strip()
                if generated_text:
                    return generated_text
        except Exception:
            return None

        return None

    def _is_valid_movement_line(self, text: str) -> bool:
        text = text.lower()

        keywords = [
            "ficha",
            "fichaje",
            "fichar",
            "firma",
            "signed",
            "signs",
            "bench",
            "benched",
            "banquillo",
            "adquiere",
            "acquire",
            "acquired",
            "traspaso",
            "transfer",
            "vende",
            "venta",
            "buy",
            "compra",
            "comprar",
            "swap",
            "intercambia",
            "join",
            "joins",
            "leaves",
            "leaves the team",
            "abandona",
            "sale",
            "renueva",
            "renew",
            "promueve",
            "promoted",
            "sube al roster",
            "entra",
            "incorpora",
            "remove",
            "removed",
            "release",
            "released",
            "waive",
            "waived",
        ]

        return any(keyword in text for keyword in keywords)

    def _unique_movements_by_date(self, movements: list[str]) -> list[str]:
        seen_dates = set()
        unique = []

        for movement in movements:
            date = self._extract_date_from_text(movement)

            # si no tiene fecha, usar texto
            key = date if date else movement

            if key not in seen_dates:
                seen_dates.add(key)
                unique.append(movement)

        return unique

    def _build_tournament_response(self, question: str) -> str | None:
        record = self._find_team_record(question)
        if not record:
            return None

        tournaments = [t for t in (record.get("tournaments") or []) if isinstance(t, str) and t.strip()]
        tournaments = self._unique_preserve_order(tournaments)
        if not tournaments:
            return f"No encontré torneos asociados a {record.get('name')} en los datos disponibles."

        count = self._parse_requested_count(question)
        selected = tournaments[:count]
        if not selected:
            return None

        if len(selected) == 1:
            return f"El próximo torneo de {record.get('name')} es {selected[0]}"

        return (
            f"Los próximos {len(selected)} torneos de {record.get('name')} son:\n"
            + "\n".join(f"- {t}" for t in selected)
        )

    def _build_participant_response(self, question: str) -> str | None:
        tournament = self._extract_tournament_name_from_question(question)
        if not tournament:
            return None

        participants = []
        for record in self.records:
            for t in (record.get("tournaments") or []) or []:
                if isinstance(t, str) and tournament.lower() in t.lower():
                    participants.append(str(record.get("name") or "Desconocido"))
                    break

        participants = self._unique_preserve_order(participants)
        if not participants:
            return f"No encontré ningún equipo participante para el torneo '{tournament}'"

        if len(participants) == 1:
            return f"El único equipo participante encontrado para {tournament} es {participants[0]}"

        return (
            f"Los equipos participantes en {tournament} son:\n"
            + "\n".join(f"- {team}" for team in participants)
        )

    def _build_roster_response(self, question: str) -> str | None:
        record = self._find_team_record(question)
        if not record:
            return None

        team_name = str(record.get("name") or "Equipo")

        movements = [
            m for m in (record.get("movements") or [])
            if isinstance(m, str) and m.strip()
        ]

        if not movements:
            return f"No encontré movimientos para {team_name}"

        # ordenar cronológicamente
        movements.sort(
            key=lambda m: self._parse_movement_date(m) or (0, 0, 0)
        )

        joined = {}   # {normalized_name: original_name}
        left = set()  # set(normalized_name)

        for movement in movements:

            parts = re.split(r"\s+y\s+|\s+and\s+|;", movement)

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                movement_clean = self._clean_movement_text(part)

                players = self._extract_players_from_movement(movement_clean)

                # ---- SWAP HANDLING ----
                joined_players, left_players = self._handle_swap_movement(
                    movement_clean,
                    team_name
                )

                if joined_players or left_players:

                    for player in joined_players:
                        joined[player.lower()] = player

                    for player in left_players:
                        left.add(player.lower())

                    continue

                # ---- NORMAL JOIN ----
                if self._is_join_movement(movement_clean):

                    for player in players:
                        normalized = player.lower()

                        if normalized not in joined:
                            joined[normalized] = player

                # ---- NORMAL LEAVE ----
                if self._is_leave_movement(movement_clean):

                    for player in players:
                        left.add(player.lower())

        # construir roster final
        roster = [
            original_name
            for normalized_name, original_name in joined.items()
            if normalized_name not in left
        ]

        roster = roster[-5:]  # últimos 5

        if not roster:
            return f"No pude reconstruir la plantilla actual de {team_name}"

        return (
            f"Plantilla actual de {team_name}:\n"
            + "\n".join(
                f"{i}. {player}"
                for i, player in enumerate(roster, start=1)
            )
        )

    def _build_generic_team_info(self, question: str) -> str | None:
        record = self._find_team_record(question)
        if not record:
            return None

        full_text = str(record.get("full_text") or record.get("combined_text") or "").strip()
        if not full_text:
            return None

        return f"Información de {record.get('name')}: {full_text}"

    def load_persisted_data(self) -> bool:
        if self.vector_store.documents:
            self.records = self._load_records()
            return True

        if not self.store_path.exists():
            pipeline_ingest = DataIngestionPipeline()
            pipeline_ingest.ingest_batch(DEFAULT_TEAM_URLS)

        self.vector_store = SimplePersistentVectorStore.load(self.store_path)
        self.records = self._load_records()
        return bool(self.vector_store.documents)

    def _build_prompt(self, question: str, context_docs: list[dict]) -> str:
        context_lines = []
        for index, doc in enumerate(context_docs, start=1):
            section = (doc.get("metadata") or {}).get("section", "resumen")
            context_lines.append(
                f"[{index}] {doc['name']} | seccion={section} | {doc['url']}: {doc['text'][:1400]}"
            )

        context = "\n\n".join(context_lines) if context_lines else "No hay contexto disponible."
        return (
            "Eres un asistente en español sobre equipos de Counter-Strike y su historial en Liquipedia. "
            "Responde solo con información apoyada en el contexto. Si no hay suficiente evidencia, dilo claramente.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n"
            "Respuesta:"
        )

    def _is_join_movement(self, text: str) -> bool:
        text = text.lower()

        keywords = [
            "acquired",
            "signed",
            "joins",
            "joined",
            "ficha",
            "fichado",
            "incorpora",
            "adquiere",
            "promoted",
            "added",
            "compra",
            "recruit",
            "recruited",
        ]

        return any(k in text for k in keywords)

    def _is_leave_movement(self, text: str) -> bool:
        text = text.lower()

        keywords = [
            "benched",
            "removed",
            "released",
            "left",
            "leaves",
            "departs",
            "transferred",
            "sold",
            "waived",
            "baja",
            "abandona",
            "vende",
            "traspasado",
            "bench",
            "banquillo",
            "separa",
        ]

        return any(k in text for k in keywords)

    def _is_roster_question(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "plantilla",
            "roster",
            "alineacion",
            "alineación",
            "lineup",
            "jugadores",
        ]

        return any(k in q for k in keywords)

    def _extract_players_from_movement(self, text: str) -> list[str]:
        blacklist = {
            "finales",
            "separa",
            "acquired",
            "signed",
            "joins",
            "benched",
            "released",
            "removed",
            "este",
            "vencimiento",
            "hasta",
            "abril",
            "asistenete",
            "hades",
            "bleed",
            "asistente",
        }

        team_names = {
            str(record.get("name") or "").lower().strip()
            for record in self.records
            if record.get("name")
        }

        forbidden_roles = [
            "entrenador",
            "coach",
            "analista",
            "assistant coach",
            "assistant",
            "manager",
            "streamer",
            "creator",
        ]

        candidates = re.findall(r"\b[A-Za-z0-9\-_]{3,20}\b", text)
        players = []

        for candidate in candidates:
            c = candidate.lower()

            if c in blacklist:
                continue
            if any(c == team.lower() for team in team_names):
                continue
            if c.isdigit():
                continue

            role_pattern = re.compile(
                rf"{re.escape(candidate)}\s+como\s+([a-zA-Záéíóúñü\s]+)",
                flags=re.IGNORECASE
            )

            role_match = role_pattern.search(text)

            if role_match:

                role_text = role_match.group(1).lower()

                if any(role in role_text for role in forbidden_roles):
                    continue

            english_role_pattern = re.compile(
                rf"{re.escape(candidate)}\s+as\s+([a-zA-Z\s]+)",
                flags=re.IGNORECASE
            )

            english_match = english_role_pattern.search(text)

            if english_match:

                role_text = english_match.group(1).lower()

                if any(role in role_text for role in forbidden_roles):
                    continue

            players.append(candidate)

        return players

    def _handle_swap_movement(
        self,
        movement: str,
        team_name: str
    ) -> tuple[list[str], list[str]]:

        movement_lower = movement.lower()

        swap_keywords = [
            "intercambian",
            "swap",
            "swapped",
            "exchange",
        ]

        if not any(k in movement_lower for k in swap_keywords):
            return [], []

        players = self._extract_players_from_movement(movement)

        if len(players) != 2:
            return [], []

        player_a, player_b = players

        normalized_team = self._normalize_text(team_name)

        return [player_b], [player_a]

    def _is_tournament_question(self, question: str) -> bool:
        q = question.lower()
        keywords = ["torneo", "torneos", "championship", "major", "iem", "blast", "pgl", "esl"]
        return any(keyword in q for keyword in keywords)

    def _build_tournament_query(self, question: str) -> str:
        return f"{question} torneos resultados partidos recientes pgl iem blast esl major"

    def _extract_latest_tournament(self, context_docs: list[dict]) -> str | None:
        if not context_docs:
            return None

        pattern = re.compile(
            r"(PGL\s+[A-Za-zÁÉÍÓÚÜÑ0-9 .:-]{2,100}(?:\d{4})?|"
            r"IEM\s+[A-Za-zÁÉÍÓÚÜÑ0-9 .:-]{2,100}(?:\d{4})?|"
            r"BLAST[A-Za-zÁÉÍÓÚÜÑ0-9 .:\-]{2,120}(?:\d{4})?|"
            r"ESL[A-Za-zÁÉÍÓÚÜÑ0-9 .:\-]{2,120}(?:\d{4})?|"
            r"CCT[A-Za-zÁÉÍÓÚÜÑ0-9 .:\-]{2,120}(?:\d{4})?|"
            r"[A-Za-zÁÉÍÓÚÜÑ0-9 .:\-]{2,80}Major\s*(?:\d{4})?)",
            flags=re.IGNORECASE,
        )

        candidates: list[tuple[int, str, str, float]] = []
        for doc in context_docs:
            section = str((doc.get("metadata") or {}).get("section", ""))
            if section not in {"torneos_resultados", "resumen_equipo", "historia_movimientos"}:
                continue

            text = str(doc.get("text") or "")
            score = float(doc.get("score") or 0.0)
            for match in pattern.finditer(text):
                name = " ".join(match.group(0).split())
                start = max(match.start() - 50, 0)
                end = min(match.end() + 50, len(text))
                window = text[start:end]
                years = re.findall(r"20\d{2}", f"{name} {window}")
                if not years:
                    continue
                year = max(int(y) for y in years)
                candidates.append((year, name, section, score))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], 1 if item[2] == "torneos_resultados" else 0, item[3]),
            reverse=True,
        )
        year, name, _, _ = candidates[0]
        clean_name = re.sub(r"\s{2,}", " ", name).strip(" .:-")
        return f"Según los datos recuperados, el torneo más reciente mencionado es {clean_name} ({year})"

    def answer(self, question: str) -> str:
        if self.generator is None:
            raise RuntimeError("El modelo local no está disponible; no se puede responder sin LLM.")

        if self._is_trajectory_question(question):
            trajectory_answer = self._build_trajectory_response(question)
            if trajectory_answer:
                return trajectory_answer
        roster_answer = self._build_roster_response(question) if self._is_roster_question(question) else None
        if roster_answer:
            return roster_answer
        
        movement_answer = self._build_movement_response(question) if self._is_movement_question(question) else None
        if movement_answer:
            return movement_answer

        participant_answer = self._build_participant_response(question) if self._is_participants_question(question) else None
        if participant_answer:
            return participant_answer

        if self._is_upcoming_tournament_question(question):
            tournament_answer = self._build_tournament_response(question)
            if tournament_answer:
                return tournament_answer

        generic_answer = self._build_generic_team_info(question)
        if generic_answer:
            return generic_answer

        context_docs = self.vector_store.search(question, top_k=8)
        prompt = self._build_prompt(question, context_docs)

        try:
            tokenizer = self.generator.tokenizer
            if getattr(tokenizer, "chat_template", None):
                messages = [
                    {
                        "role": "system",
                        "content": "Eres un asistente útil que responde sobre equipos de Counter-Strike en español.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            result = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                do_sample=False,
                return_full_text=False,
            )

            if isinstance(result, list) and result:
                generated_text = result[0].get("generated_text", "")
                if generated_text.strip():
                    return generated_text.strip()
        except Exception as exc:
            raise RuntimeError(f"Falló la generación con el modelo local: {exc}") from exc

        raise RuntimeError("El modelo local no devolvió una respuesta válida.")

    def run_interactive(self) -> None:
        print("Chatbot listo. Escribe 'salir' para terminar.\n")
        while True:
            try:
                question = input("Tú: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo...")
                break

            if not question:
                continue
            if question.lower() in {"salir", "exit", "quit"}:
                break

            answer = self.answer(question)
            print(f"Bot: {answer}\n")


def _ensure_store() -> None:
    if DEFAULT_STORE_FILE.exists():
        return
    pipeline_ingest = DataIngestionPipeline()
    pipeline_ingest.ingest_batch(DEFAULT_TEAM_URLS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chatbot RAG sobre equipos de Counter-Strike.")
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL))
    parser.add_argument("--store", default=str(DEFAULT_STORE_FILE))
    args = parser.parse_args()

    _ensure_store()

    chatbot = EsportsChatbot(model_name=args.model, store_path=args.store)
    if not chatbot.load_persisted_data():
        print("No hay datos persistidos disponibles para el chatbot.")
        return 1

    chatbot.run_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())