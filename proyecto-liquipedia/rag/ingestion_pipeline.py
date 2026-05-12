"""Pipeline RAG ligero basado en recuperación léxica (sin deep learning)."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_áéíóúÁÉÍÓÚñÑüÜ]+")
_STOPWORDS = {
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "eramos",
    "eran",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estaba",
    "estaban",
    "estado",
    "estamos",
    "estan",
    "estar",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "fueron",
    "ha",
    "hace",
    "hacia",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "mis",
    "mucho",
    "muy",
    "nada",
    "ni",
    "no",
    "nos",
    "nosotros",
    "o",
    "os",
    "otra",
    "otras",
    "otro",
    "otros",
    "para",
    "pero",
    "poco",
    "por",
    "porque",
    "que",
    "quien",
    "se",
    "si",
    "sin",
    "sobre",
    "su",
    "sus",
    "tambien",
    "te",
    "tiene",
    "tienen",
    "todo",
    "tu",
    "tus",
    "un",
    "una",
    "uno",
    "unas",
    "unos",
    "y",
    "ya",
}


@dataclass
class RetrievedDoc:
    """Documento recuperado con su puntuación."""

    id: str
    text: str
    metadata: Dict
    score: float


class SimplePersistentVectorStore:
    """Almacenamiento persistente con índice invertido y scoring BM25."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.documents: List[Dict] = []
        self.metadata: List[Dict] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_term_freq: List[Counter] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len: float = 0.0
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            self._ensure_parent()
            self._save()
            return

        with self.store_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.documents = payload.get("documents", [])
        self.metadata = payload.get("metadata", [])
        self._rebuild_index()

    def _ensure_parent(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        self._ensure_parent()
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"documents": self.documents, "metadata": self.metadata},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]

    def _rebuild_index(self) -> None:
        self._doc_tokens = []
        self._doc_term_freq = []
        self._doc_freq = Counter()

        for doc in self.documents:
            tokens = self._tokenize(doc.get("text", ""))
            self._doc_tokens.append(tokens)
            tf = Counter(tokens)
            self._doc_term_freq.append(tf)
            self._doc_freq.update(tf.keys())

        doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

    def count(self) -> int:
        return len(self.documents)

    def clear(self) -> None:
        self.documents = []
        self.metadata = []
        self._rebuild_index()
        self._save()

    def add_documents(self, documents: List[Dict]) -> None:
        start = len(self.documents)
        for idx, item in enumerate(documents):
            text = item.get("text", "").strip()
            if not text:
                continue
            self.documents.append(
                {
                    "id": f"doc_{start + idx}",
                    "text": text,
                    "length": len(text),
                    "ingested_at": datetime.now().isoformat(),
                }
            )
            self.metadata.append(item.get("metadata", {}))

        self._rebuild_index()
        self._save()

    def search(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.documents)
        k1 = 1.5
        b = 0.75

        scores: List[RetrievedDoc] = []
        for i, doc in enumerate(self.documents):
            tf = self._doc_term_freq[i]
            doc_len = max(1, len(self._doc_tokens[i]))
            score = 0.0

            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue

                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * (doc_len / max(self._avg_doc_len, 1.0)))
                score += idf * ((freq * (k1 + 1)) / max(denom, 1e-9))

            if score <= 0:
                continue

            scores.append(
                RetrievedDoc(
                    id=doc.get("id", f"doc_{i}"),
                    text=doc.get("text", ""),
                    metadata=self.metadata[i] if i < len(self.metadata) else {},
                    score=score,
                )
            )

        scores.sort(key=lambda d: d.score, reverse=True)
        return scores[:top_k]


class RAGEngine:
    """Motor RAG simple con recuperación y respuesta extractiva."""

    _FOUNDATION_QUERY_RE = re.compile(
        r"\b(fundad[oa]s?|fundaci[oó]n|cread[oa]s?|creaci[oó]n|found(?:ed|ing)?|founders?|when\s+was|cu[aá]ndo\s+se\s+fund[oó]|cu[aá]ndo\s+fue\s+cread[oa])\b",
        re.IGNORECASE,
    )

    _ROSTER_QUERY_RE = re.compile(
        r"\b(lista|jugadores?|plantilla|roster|alineaci[oó]n|equipo|staff|organizaci[oó]n|manager|gerente|entrenador|coach|ceo|director|analista)\b",
        re.IGNORECASE,
    )

    _STAFF_QUERY_RE = re.compile(
        r"\b(entrenador|coach|manager|gerente|ceo|director|analista|igl|leader|líder del juego|lider del juego|staff|personal)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        base = Path(__file__).parent.parent
        self.vector_store = SimplePersistentVectorStore(
            base / "chatbot" / "data" / "vector_store" / "store.json"
        )
        self.llm_model_name = os.getenv(
            "LOCAL_LLM_MODEL",
            "microsoft/Phi-3-mini-4k-instruct",
        )
        self._load_local_llm()

    def _load_local_llm(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar el LLM local '{self.llm_model_name}'. "
                "Instala o descarga el modelo Phi-3-mini y vuelve a intentarlo."
            ) from exc

    def ingest_documents(self, documents: List[Dict]) -> None:
        self.vector_store.add_documents(documents)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\[\s*e\s*\]\[\s*h\s*\]', '', text)
        text = re.sub(r'Descripción general\s+Resultados\s+Partidos', '', text)
        text = re.sub(r'Academia\s+\w+\s+\w+\s+Academia', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _top_section_text(text: str, max_chars: int = 1600) -> str:
        start_match = re.search(r"\bInformación del equipo\b", text, flags=re.IGNORECASE)
        if start_match:
            text = text[start_match.start():]

        markers = [
            r"\bContenido\b",
            r"\bCronolog[íi]a\b",
            r"\bHistoria\b",
            r"\bTimeline\b",
            r"\bLista de jugadores\b",
            r"\bPlantilla de jugadores\b",
            r"\bOrganizaci[oó]n\b",
            r"\bResults\b",
            r"\bResultados\b",
        ]
        earliest = len(text)
        for marker in markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                earliest = min(earliest, match.start())
        return text[: min(earliest, max_chars)]

    @classmethod
    def _question_is_foundation_related(cls, question: str) -> bool:
        return bool(cls._FOUNDATION_QUERY_RE.search(question or ""))

    @staticmethod
    def _token_overlap(left_text: str, right_text: str) -> int:
        left_tokens = set(SimplePersistentVectorStore._tokenize(left_text))
        right_tokens = set(SimplePersistentVectorStore._tokenize(right_text))
        return len(left_tokens & right_tokens)

    def _order_docs_for_question(self, question: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        def score(doc: RetrievedDoc) -> tuple:
            name = doc.metadata.get("name", "")
            return (
                self._token_overlap(question, name),
                round(doc.score, 6),
            )

        return sorted(docs, key=score, reverse=True)

    @staticmethod
    def _extract_field(text: str, field_names: List[str], stop_names: List[str]) -> str:
        field_group = "|".join(re.escape(name) for name in field_names)
        stop_group = "|".join(re.escape(name) for name in stop_names)
        pattern = rf"(?:{field_group})\s*:?\s*(.+?)(?=\s+(?:{stop_group})\s*:|\s+Historial\s+de\s+enlaces|\s+Contenido|\s+Cronolog[íi]a|\s+Historia|\s+Resultados|\s+Partidos|\s+Lista|\s+Organizaci[oó]n|\s+\[editar\]|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return ""
        value = match.group(1).strip()
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _extract_section_excerpt(text: str, section_names: List[str], stop_names: List[str], max_length: int = 260) -> str:
        section_group = "|".join(re.escape(name) for name in section_names)
        stop_group = "|".join(re.escape(name) for name in stop_names)
        pattern = rf"(?:{section_group})(?:\s*\[editar\])?\s*(.+?)(?=\s+(?:{stop_group})(?:\s*\[editar\])?\b|\s+Referencias|\s+REFERENCIAS\s+EXTERNAS|$)"
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return ""

        def score(excerpt: str) -> tuple:
            letters = sum(ch.isalpha() for ch in excerpt)
            digits = sum(ch.isdigit() for ch in excerpt)
            return (len(excerpt), letters - digits * 2)

        excerpts = [re.sub(r"\s+", " ", match.group(1)).strip() for match in matches]
        excerpt = max(excerpts, key=score)
        if len(excerpt) > max_length:
            excerpt = excerpt[: max_length - 3].rstrip() + "..."
        return excerpt

    def _build_foundation_response(self, docs: List[RetrievedDoc]) -> str:
        lines: List[str] = []

        for doc in docs:
            text = self._top_section_text(self._clean_text(doc.text))
            name = doc.metadata.get("name", "Fuente")

            sentences = self._split_sentences(text)
            founding_sentence = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(r"\b(fundad[oa]|cread[oa]|found(?:ed|ing)?)\b", sentence, re.IGNORECASE)
                ),
                "",
            )

            founders = self._extract_field(
                text,
                ["Fundadores", "Founders"],
                ["CEO", "Manager", "Líder del juego", "Entrenadores", "Analistas", "Ubicación", "Location", "Región", "Region"],
            )
            location = self._extract_field(
                text,
                ["Ubicación", "Location"],
                ["Región", "Region", "Fundadores", "Founders", "CEO", "Manager", "Líder del juego", "Entrenadores", "Analistas"],
            )
            created = self._extract_field(
                text,
                ["Creado", "Created", "Founded", "Fundada", "Fundado"],
                ["Disuelto", "Dissolved", "Fundadores", "Founders", "CEO", "Manager", "Ubicación", "Location", "Región", "Region"],
            )

            parts: List[str] = []
            if founding_sentence:
                parts.append(founding_sentence)
            elif created:
                parts.append(f"Se creó/fundó en {created}.")

            if founders:
                parts.append(f"Fundadores: {founders}.")

            if location:
                parts.append(f"Ubicación: {location}.")

            if not parts:
                continue

            lines.append(f"- {name}: {' '.join(parts)}")

        if not lines:
            return (
                "Recuperé documentos, pero no pude extraer una fecha de fundación clara. "
                "Prueba con el nombre exacto del equipo o vuelve a formular la pregunta."
            )

        return "📚 Según la información recuperada de Liquipedia:\n" + "\n".join(lines[:3])

    def _build_staff_response(self, docs: List[RetrievedDoc]) -> str:
        lines: List[str] = []

        for doc in docs:
            text = self._top_section_text(self._clean_text(doc.text))
            name = doc.metadata.get("name", "Fuente")

            coach = self._extract_field(
                text,
                ["Entrenadores", "Coach", "Coaches"],
                ["Analistas", "Manager", "Gerente", "Líder del juego", "Leader", "Ubicación", "Location", "Región", "Region", "CEO"],
            )
            manager = self._extract_field(
                text,
                ["Gerente", "Manager", "Director del equipo", "Team Manager"],
                ["Entrenadores", "Coach", "Analistas", "CEO", "Ubicación", "Location", "Región", "Region"],
            )
            ceo = self._extract_field(
                text,
                ["CEO", "Director ejecutivo", "Chief Executive Officer"],
                ["Gerente", "Manager", "Entrenadores", "Coach", "Analistas", "Ubicación", "Location", "Región", "Region"],
            )
            analyst = self._extract_field(
                text,
                ["Analistas", "Analista", "Analyst"],
                ["CEO", "Gerente", "Manager", "Entrenadores", "Coach", "Ubicación", "Location", "Región", "Region"],
            )
            igl = self._extract_field(
                text,
                ["Líder del juego", "Lider del juego", "In-game leader", "IGL"],
                ["CEO", "Gerente", "Manager", "Entrenadores", "Coach", "Analistas", "Ubicación", "Location", "Región", "Region"],
            )

            pieces: List[str] = []
            if coach:
                pieces.append(f"Entrenador: {coach}.")
            if manager:
                pieces.append(f"Manager: {manager}.")
            if ceo:
                pieces.append(f"CEO: {ceo}.")
            if analyst:
                pieces.append(f"Analista: {analyst}.")
            if igl:
                pieces.append(f"Líder del juego: {igl}.")

            if not pieces:
                continue

            lines.append(f"- {name}: {' '.join(pieces)}")

        if not lines:
            return (
                "No pude leer bien el bloque de personal del equipo. "
                "Prueba con una pregunta más concreta, por ejemplo: 'quién es el coach de Vitality' o 'quién es el CEO de G2'."
            )

        return "📚 Según la información recuperada de Liquipedia:\n" + "\n".join(lines[:3])

    def _build_roster_response(self, docs: List[RetrievedDoc]) -> str:
        lines: List[str] = []

        for doc in docs:
            text = self._clean_text(doc.text)
            name = doc.metadata.get("name", "Fuente")

            roster_excerpt = self._extract_section_excerpt(
                text,
                ["Lista de jugadores", "Plantilla de jugadores", "Lista", "Roster"],
                ["Organización", "Resultados", "Galería", "Logotipos", "Referencias"],
            )
            roster_clause = ""
            if roster_excerpt:
                clause_match = re.search(
                    r"(?:consta de|is composed of|comprende|formada por|compuesta por)\s+([^.;]+)",
                    roster_excerpt,
                    flags=re.IGNORECASE,
                )
                if clause_match:
                    roster_clause = re.sub(r"\s+", " ", clause_match.group(1)).strip()
            org_excerpt = self._extract_section_excerpt(
                text,
                ["Organización", "Organization"],
                ["Resultados", "Galería", "Logotipos", "Referencias"],
            )

            pieces: List[str] = []
            if roster_clause:
                pieces.append(f"Jugadores actuales: {roster_clause}.")
            elif roster_excerpt:
                pieces.append(f"Plantilla: {roster_excerpt}")
            if org_excerpt:
                pieces.append(f"Organización: {org_excerpt}")

            if not pieces:
                continue

            lines.append(f"- {name}: {' '.join(pieces)}")

        if not lines:
            return (
                "Recuperé los documentos, pero no pude leer bien la tabla o la sección solicitada. "
                "Prueba con una pregunta más concreta, por ejemplo: 'quién es el coach de G2' o 'qué jugadores tiene Vitality'."
            )

        return "📚 Según la información recuperada de Liquipedia:\n" + "\n".join(lines[:3])

    @staticmethod
    def _format_context(docs: List[RetrievedDoc], max_chars_per_doc: int = 2200) -> str:
        blocks: List[str] = []
        for doc in docs:
            name = doc.metadata.get("name", "Fuente")
            url = doc.metadata.get("url", "")
            text = RAGEngine._top_section_text(RAGEngine._clean_text(doc.text), max_chars=max_chars_per_doc)
            if len(text) > max_chars_per_doc:
                text = text[: max_chars_per_doc - 3].rstrip() + "..."
            header = f"EQUIPO: {name}"
            if url:
                header += f" | URL: {url}"
            blocks.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def _generate_llm_answer(self, question: str, docs: List[RetrievedDoc]) -> str:
        context = self._format_context(docs)
        system_prompt = (
            "Eres un asistente en español especializado en equipos de esports de Liquipedia. "
            "Responde solo con la información presente en el contexto recuperado. "
            "Si el contexto no contiene la respuesta, dilo de forma breve y no inventes datos. "
            "Cuando la pregunta sea sobre fundación, staff, roster o organización, usa el texto del bloque superior de la página y resume con claridad."
        )
        user_prompt = (
            f"Pregunta del usuario: {question}\n\n"
            f"Contexto recuperado de Liquipedia:\n{context}\n\n"
            "Responde de forma breve, clara y en español. Incluye solo los datos relevantes."
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nRespuesta:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_length:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return content or "No pude generar una respuesta válida con el LLM local."

    @staticmethod
    def _sentence_score(sentence: str, query_tokens: List[str]) -> int:
        low = sentence.lower()
        return sum(1 for tok in query_tokens if tok in low)

    def _build_response(self, question: str, docs: List[RetrievedDoc]) -> str:
        docs = self._order_docs_for_question(question, docs)
        return self._generate_llm_answer(question, docs)

    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        docs = self.vector_store.search(question, top_k=top_k)
        response = self._build_response(question, docs)
        return {
            "question": question,
            "response": response,
            "sources": [
                {
                    "id": d.id,
                    "score": round(d.score, 4),
                    "metadata": d.metadata,
                }
                for d in docs
            ],
        }
