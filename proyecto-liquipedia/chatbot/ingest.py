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

try:
    from scraper.main_scraper import LiquipediaTeamScraper
except Exception:
    LiquipediaTeamScraper = None

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


QUERY_EXPANSIONS: dict[str, list[str]] = {
    "fichaje": ["fichajes", "fichar", "ficha", "adquiere", "transferencia", "movimiento", "movimientos", "venta", "roster", "join", "bench"],
    "fichajes": ["fichaje", "fichar", "ficha", "adquiere", "transferencia", "movimiento", "movimientos", "venta", "roster", "join", "bench"],
    "movimiento": ["movimiento", "movimientos", "fichaje", "fichajes", "fichar", "venta", "transferencia", "bench", "benching"],
    "movimientos": ["movimiento", "movimientos", "fichaje", "fichajes", "fichar", "venta", "transferencia", "bench", "benching"],
    "staff": ["organizacion", "organización", "manager", "coach", "entrenador", "entrenadores", "analista"],
    "entrenador": ["coach", "staff", "organizacion", "organización"],
    "torneo": ["torneos", "resultados", "partidos", "logros", "upcoming"],
    "torneos": ["torneo", "resultados", "partidos", "logros", "upcoming"],
}


def _expand_query_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(QUERY_EXPANSIONS.get(token, []))
    # Deduplicar conservando orden.
    return list(dict.fromkeys(expanded))


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
        query_tokens = _expand_query_tokens(_tokenize(query))
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

    def _record_url(self, record: dict) -> str:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        return str(record.get("url") or metadata.get("url") or "")

    def _load_records_from_file(self, file_path: str | Path) -> list[dict]:
        try:
            with Path(file_path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, ValueError, OSError):
            return []

        if isinstance(payload, dict):
            records = payload.get("documents", []) or payload.get("items", [])
        else:
            records = payload

        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _section_from_heading(self, heading: str) -> str:
        h = heading.lower()
        if "organiz" in h or "staff" in h:
            return "staff_organizacion"
        if "jugador" in h or "plantilla" in h or "roster" in h:
            return "jugadores_fichajes"
        if "resultado" in h or "partido" in h or "logro" in h or "torneo" in h:
            return "torneos_resultados"
        if "historia" in h or "cronolog" in h:
            return "historia_movimientos"
        if "referencia" in h:
            return "referencias"
        return "resumen_equipo"

    def _split_sections(self, raw_text: str) -> list[tuple[str, str]]:
        text = " ".join((raw_text or "").split())
        if len(text) < 180:
            return [("resumen_equipo", text)] if text else []

        heading_pattern = re.compile(
            r"\b(Próximos torneos|Próximos partidos|Historia|Cronología|Lista de jugadores|Plantilla de jugadores|Organización|Resultados|Partidos recientes|Logros|Referencias)\b",
            flags=re.IGNORECASE,
        )
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [("resumen_equipo", text)]

        sections: list[tuple[str, str]] = []
        start = 0
        current_label = "resumen_equipo"
        for match in matches:
            if match.start() - start >= 120:
                sections.append((current_label, text[start:match.start()].strip()))
            current_label = self._section_from_heading(match.group(1))
            start = match.start()

        tail = text[start:].strip()
        if len(tail) >= 120:
            sections.append((current_label, tail))

        chunked: list[tuple[str, str]] = []
        max_chars = 1400
        overlap = 220
        for label, section_text in sections:
            if len(section_text) <= max_chars:
                chunked.append((label, section_text))
                continue

            cursor = 0
            while cursor < len(section_text):
                chunk = section_text[cursor : cursor + max_chars].strip()
                if len(chunk) >= 120:
                    chunked.append((label, chunk))
                if cursor + max_chars >= len(section_text):
                    break
                cursor += max_chars - overlap

        return chunked

    def _normalize_record(self, record: dict) -> list[dict]:
        raw_text = (
            record.get("text")
            or record.get("content")
            or record.get("combined_text")
            or record.get("full_text")
            or ""
        )

        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("name", record.get("name") or metadata.get("name") or "Desconocido")
        metadata.setdefault("url", record.get("url") or metadata.get("url") or "")

        documents: list[dict] = []
        movements = record.get("movements", [])
        if movements:
            for movement_index, movement_line in enumerate(movements, start=1):
                line = str(movement_line or "").strip()
                if not line:
                    continue

                movement_text = f"{metadata['name']} {line}"
                cleaned_movement_text = self.cleaner.clean_text(movement_text)
                if not validate_text_quality(cleaned_movement_text, min_length=30):
                    continue

                movement_metadata = dict(metadata)
                movement_metadata["section"] = "movimientos"
                movement_metadata["chunk_index"] = movement_index
                movement_metadata["movement_text"] = line

                documents.append(
                    {
                        "name": metadata["name"],
                        "url": metadata["url"],
                        "text": movement_text,
                        "clean_text": cleaned_movement_text,
                        "metadata": movement_metadata,
                    }
                )

        for chunk_index, (section, section_text) in enumerate(self._split_sections(raw_text), start=1):
            cleaned_text = self.cleaner.clean_text(section_text)
            if not validate_text_quality(cleaned_text, min_length=60):
                continue

            chunk_metadata = dict(metadata)
            chunk_metadata["section"] = section
            chunk_metadata["chunk_index"] = chunk_index

            documents.append(
                {
                    "name": metadata["name"],
                    "url": metadata["url"],
                    "text": section_text,
                    "clean_text": cleaned_text,
                    "metadata": chunk_metadata,
                }
            )

        if documents:
            return documents

        cleaned_text = self.cleaner.clean_text(raw_text)
        if not validate_text_quality(cleaned_text, min_length=60):
            return []

        fallback_metadata = dict(metadata)
        fallback_metadata["section"] = "resumen_equipo"
        fallback_metadata["chunk_index"] = 1
        return [
            {
                "name": metadata["name"],
                "url": metadata["url"],
                "text": raw_text,
                "clean_text": cleaned_text,
                "metadata": fallback_metadata,
            }
        ]

    def _save_records(self, records: Sequence[dict]) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with self.data_file.open("w", encoding="utf-8") as handle:
            json.dump(list(records), handle, ensure_ascii=False, indent=2)

    def _scrape_missing_records(self, team_urls: Sequence[str]) -> list[dict]:
        if not team_urls or LiquipediaTeamScraper is None:
            return []

        scraper = LiquipediaTeamScraper()
        scraped_records: list[dict] = []
        try:
            for url in team_urls:
                try:
                    record = scraper.scrape_team_page(url)
                except Exception:
                    record = None
                if record:
                    scraped_records.append(record)
        finally:
            scraper.cleanup()

        return scraped_records

    def _load_local_records(self) -> list[dict]:
        if self.data_file.exists():
            return self._load_records_from_file(self.data_file)
        return []

    def ingest_batch(self, team_urls: Sequence[str] | None = None) -> list[dict]:
        requested_urls = set(team_urls or [])
        records = self._load_local_records()

        if requested_urls:
            filtered = [record for record in records if self._record_url(record) in requested_urls]
            present_urls = {self._record_url(record) for record in filtered if self._record_url(record)}
            missing_urls = sorted(url for url in requested_urls if url not in present_urls)

            scraped = self._scrape_missing_records(missing_urls)
            if scraped:
                by_url = {self._record_url(record): record for record in records if self._record_url(record)}
                for record in scraped:
                    record_url = self._record_url(record)
                    if record_url:
                        by_url[record_url] = record
                records = list(by_url.values())
                self._save_records(records)

                filtered = [record for record in records if self._record_url(record) in requested_urls]

            if filtered:
                records = filtered

        normalized_documents = []
        for record in records:
            normalized_documents.extend(self._normalize_record(record))

        store = SimplePersistentVectorStore([])
        store.add_documents(normalized_documents)
        store.save(self.store_file)
        return normalized_documents

    def cleanup(self) -> None:
        return None
