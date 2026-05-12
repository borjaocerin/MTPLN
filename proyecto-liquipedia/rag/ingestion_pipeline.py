"""Pipeline RAG ligero basado en recuperación léxica (sin deep learning)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


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

    def __init__(self):
        base = Path(__file__).parent.parent
        self.vector_store = SimplePersistentVectorStore(
            base / "chatbot" / "data" / "vector_store" / "store.json"
        )

    def ingest_documents(self, documents: List[Dict]) -> None:
        self.vector_store.add_documents(documents)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

    @staticmethod
    def _sentence_score(sentence: str, query_tokens: List[str]) -> int:
        low = sentence.lower()
        return sum(1 for tok in query_tokens if tok in low)

    def _build_response(self, question: str, docs: List[RetrievedDoc]) -> str:
        if not docs:
            return (
                "No encontré información suficiente en la base local. "
                "Prueba con otro equipo/jugador o vuelve a ejecutar ingest.py."
            )

        query_tokens = SimplePersistentVectorStore._tokenize(question)
        candidate_lines: List[str] = []

        for rank, doc in enumerate(docs, start=1):
            name = doc.metadata.get("name", "Fuente")
            url = doc.metadata.get("url", "")
            
            # Limpiar artefactos comunes del scraping
            text = doc.text
            text = re.sub(r'\[\s*e\s*\]\[\s*h\s*\]', '', text)  # Remover [e][h]
            text = re.sub(r'Descripción general\s+Resultados\s+Partidos', '', text)
            text = re.sub(r'Academia\s+\w+\s+\w+\s+Academia', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            sentences = self._split_sentences(text)
            if not sentences:
                continue

            scored = sorted(
                sentences,
                key=lambda s: self._sentence_score(s, query_tokens),
                reverse=True,
            )
            
            # Tomar las 2 mejores sentencias y combinarlas
            best_sentences = [s.strip() for s in scored[:2] if len(s.strip()) > 20]
            if not best_sentences:
                continue
                
            best = " ".join(best_sentences)
            if len(best) > 200:
                best = best[:197] + "..."

            source = f"{name}" if not url else f"{name} ({url})"
            candidate_lines.append(f"{rank}. {best}\n   [Fuente: {source}]")

        if not candidate_lines:
            return (
                "Recuperé documentos, pero no pude sintetizar una respuesta clara. "
                "Intenta con una pregunta más específica."
            )

        intro = "📚 Según la información recuperada de Liquipedia:\n"
        return intro + "\n".join(candidate_lines[:2])

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
