import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scraper.cleaner import TextCleaner, validate_text_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "latest_ingest.json"
DEFAULT_STORE_FILE = Path(__file__).resolve().parent / "data" / "vector_store" / "store.json"


def _load_default_team_urls() -> list[str]:
    sources_file = PROJECT_ROOT / "sources_example.json"
    if not sources_file.exists():
        return []

    try:
        with sources_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    teams = payload.get("teams", [])
    return [team for team in teams if isinstance(team, str)]


DEFAULT_TEAM_URLS = _load_default_team_urls()


def load_sources_file(sources_file: str | Path) -> list[str]:
    with Path(sources_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    teams = payload.get("teams", [])
    if not isinstance(teams, list):
        raise ValueError("El archivo de fuentes debe contener una lista 'teams'.")
    return [team for team in teams if isinstance(team, str)]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wáéíóúñü]+", text.lower(), flags=re.UNICODE)


class SimplePersistentVectorStore:
    def __init__(self, documents: list[dict] | None = None):
        self.documents: list[dict] = documents or []
        self._avg_doc_len = 0.0
        self._document_frequency: dict[str, int] = {}
        self._refresh_statistics()

    def _refresh_statistics(self) -> None:
        total_length = 0
        document_frequency = defaultdict(int)

        for document in self.documents:
            tokens = document.get("tokens", [])
            total_length += len(tokens)
            for token in set(tokens):
                document_frequency[token] += 1

        self._avg_doc_len = (total_length / len(self.documents)) if self.documents else 0.0
        self._document_frequency = dict(document_frequency)

    @classmethod
    def load(cls, path: str | Path) -> "SimplePersistentVectorStore":
        path = Path(path)
        if not path.exists():
            return cls([])

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        documents = payload.get("documents", [])
        for document in documents:
            counts = document.get("term_counts", {})
            if isinstance(counts, dict):
                document["term_counts"] = {str(token): int(count) for token, count in counts.items()}
            document["tokens"] = [str(token) for token in document.get("tokens", [])]

        return cls(documents)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "documents": self.documents,
            "document_count": len(self.documents),
            "avg_doc_len": self._avg_doc_len,
            "document_frequency": self._document_frequency,
        }

        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def add_documents(self, documents: Sequence[dict]) -> None:
        for document in documents:
            text = document.get("clean_text") or document.get("text") or document.get("full_text") or ""
            tokens = _tokenize(text)
            prepared = dict(document)
            prepared["text"] = text
            prepared["tokens"] = tokens
            prepared["term_counts"] = dict(Counter(tokens))
            prepared["token_count"] = len(tokens)
            self.documents.append(prepared)

        self._refresh_statistics()

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores: list[tuple[float, dict]] = []
        total_documents = len(self.documents)
        k1 = 1.5
        b = 0.75

        for document in self.documents:
            counts = document.get("term_counts", {})
            doc_len = max(int(document.get("token_count", len(document.get("tokens", [])))), 1)
            score = 0.0
            for token in query_tokens:
                tf = int(counts.get(token, 0))
                if tf == 0:
                    continue
                df = self._document_frequency.get(token, 0)
                if df == 0:
                    continue
                idf = math.log(1 + ((total_documents - df + 0.5) / (df + 0.5)))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1.0))
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((score, document))

        scores.sort(key=lambda item: item[0], reverse=True)
        ranked = []
        for score, document in scores[:top_k]:
            ranked.append(
                {
                    "score": score,
                    "name": document.get("name") or document.get("metadata", {}).get("name") or "Desconocido",
                    "url": document.get("url") or document.get("metadata", {}).get("url") or "",
                    "text": document.get("text", ""),
                    "metadata": document.get("metadata", {}),
                }
            )
        return ranked


class DataIngestionPipeline:
    def __init__(self, data_file: str | Path | None = None, store_file: str | Path | None = None):
        self.data_file = Path(data_file) if data_file else DEFAULT_DATA_FILE
        self.store_file = Path(store_file) if store_file else DEFAULT_STORE_FILE
        self.cleaner = TextCleaner()

    def _load_records_from_file(self, file_path: str | Path) -> list[dict]:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            records = payload.get("documents", []) or payload.get("items", [])
        else:
            records = payload

        if not isinstance(records, list):
            raise ValueError("El archivo de ingestión no contiene una lista válida de registros.")
        return [record for record in records if isinstance(record, dict)]

    def _normalize_record(self, record: dict) -> dict | None:
        raw_text = (
            record.get("text")
            or record.get("content")
            or record.get("combined_text")
            or record.get("full_text")
            or ""
        )
        cleaned_text = self.cleaner.clean_text(raw_text)
        if not validate_text_quality(cleaned_text, min_length=80):
            return None

        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("name", record.get("name") or metadata.get("name") or "Desconocido")
        metadata.setdefault("url", record.get("url") or metadata.get("url") or "")

        return {
            "name": metadata["name"],
            "url": metadata["url"],
            "text": raw_text,
            "clean_text": cleaned_text,
            "metadata": metadata,
        }

    def _load_local_records(self) -> list[dict]:
        if self.data_file.exists():
            return self._load_records_from_file(self.data_file)
        return []

    def ingest_batch(self, team_urls: Sequence[str] | None = None) -> list[dict]:
        requested_urls = set(team_urls or [])
        records = self._load_local_records()

        if requested_urls:
            filtered = [record for record in records if record.get("url") in requested_urls]
            if filtered:
                records = filtered

        normalized_documents = []
        for record in records:
            normalized = self._normalize_record(record)
            if normalized is not None:
                normalized_documents.append(normalized)

        store = SimplePersistentVectorStore([])
        store.add_documents(normalized_documents)
        store.save(self.store_file)
        return normalized_documents

    def cleanup(self) -> None:
        return None
