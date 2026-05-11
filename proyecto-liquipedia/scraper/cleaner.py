import json
import re
from typing import Iterable


class TextCleaner:
    """Limpieza básica de texto para facilitar la recuperación en el chatbot."""

    _NOISE_PATTERNS = [
        r"\[\d+\]",  # referencias tipo [1], [23]
        r"\s+",  # espacios repetidos
    ]

    def __init__(self):
        self.blacklist = [
            "cookie",
            "política de privacidad",
            "privacy policy",
            "utiliza cookies",
            "cloudflare",
            "all rights reserved",
            "newsletter",
            "configuración de cookies",
        ]

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text.strip()
        for pattern in self._NOISE_PATTERNS:
            replacement = " " if pattern == r"\s+" else ""
            cleaned = re.sub(pattern, replacement, cleaned)

        # Quitar segmentos de ruido muy frecuentes.
        low = cleaned.lower()
        for word in self.blacklist:
            if word in low:
                cleaned = re.sub(re.escape(word), "", cleaned, flags=re.IGNORECASE)
                low = cleaned.lower()

        return cleaned.strip()


def validate_text_quality(text: str, min_length: int = 50) -> bool:
    """Valida si un texto tiene contenido útil para indexar."""
    if not text or len(text.strip()) < min_length:
        return False

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    total_chars = len(text)
    if total_chars == 0:
        return False

    alpha_ratio = alpha_chars / total_chars
    return alpha_ratio >= 0.45


def _contains_any(text: str, words: Iterable[str]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def limpiar_datos(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Registros originales: {len(data)}")

    blacklist = [
        "cookie",
        "política de privacidad",
        "privacy policy",
        "utiliza cookies",
        "maliciosos",
        "verificar que no eres un bot",
        "cloudflare",
        "derechos reservados",
        "all rights reserved",
        "18 years of age",
        "newsletter",
        "suscríbete",
        "configuración de cookies",
        "mantenimiento",
    ]

    keywords_fichajes = [
        "join",
        "welcome",
        "fichaje",
        "departure",
        "farewell",
        "roster",
        "contract",
        "free agent",
        "lft",
        "bench",
        "replaces",
        "signing",
        "bienvenida",
        "despedida",
        "contrato",
        "agente libre",
    ]

    cleaned_data = []

    for entry in data:
        texto = entry.get("content")
        if not texto or len(texto) < 45:
            continue
        if _contains_any(texto, blacklist):
            continue
        if _contains_any(texto, keywords_fichajes):
            cleaned_data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"Registros después de la limpieza: {len(cleaned_data)}")
    print(f"Archivo guardado como: {output_file}")


if __name__ == "__main__":
    limpiar_datos("external_refs_multiple_es.json", "fichajes_limpios.json")