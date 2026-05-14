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
        match = re.search(r"\b(\d+)\b", question)
        if not match:
            return 1
        return max(1, int(match.group(1)))

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

        movements = [m for m in (record.get("movements") or []) if isinstance(m, str) and m.strip()]
        if not movements:
            return f"No encontré movimientos registrados para {record.get('name')} en los datos disponibles."

        count = self._parse_requested_count(question)
        selected = movements[-count:]
        selected = self._unique_preserve_order(selected)

        lines = []
        for movement in selected:
            date = self._extract_date_from_text(movement)
            if date:
                description = movement.replace(date, "").strip(" -:;,. ")
                lines.append(f"{date}: {description}")
            else:
                lines.append(movement)

        if len(lines) == 1:
            return f"El último movimiento de {record.get('name')} es: {lines[0]}"

        return (
            f"Los últimos {len(lines)} movimientos de {record.get('name')} son:\n"
            + "\n".join(f"- {line}" for line in lines)
        )

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
            return f"El próximo torneo de {record.get('name')} es {selected[0]}."

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
            return f"No encontré ningún equipo participante para el torneo '{tournament}'."

        if len(participants) == 1:
            return f"El único equipo participante encontrado para {tournament} es {participants[0]}."

        return (
            f"Los equipos participantes en {tournament} son:\n"
            + "\n".join(f"- {team}" for team in participants)
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
        return f"Según los datos recuperados, el torneo más reciente mencionado es {clean_name} ({year})."

    def answer(self, question: str) -> str:
        if self.generator is None:
            raise RuntimeError("El modelo local no está disponible; no se puede responder sin LLM.")

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