import argparse
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
    from chatbot.ingest import DataIngestionPipeline, DEFAULT_STORE_FILE, DEFAULT_TEAM_URLS, SimplePersistentVectorStore
else:
    from .ingest import DataIngestionPipeline, DEFAULT_STORE_FILE, DEFAULT_TEAM_URLS, SimplePersistentVectorStore


class EsportsChatbot:
    def __init__(self, model_name: str | None = None, store_path: str | Path | None = None):
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL)
        self.max_new_tokens = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "96"))
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_FILE
        self.vector_store = SimplePersistentVectorStore.load(self.store_path)
        self.generator = self._load_generator()

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

    def load_persisted_data(self) -> bool:
        if self.vector_store.documents:
            return True

        if not self.store_path.exists():
            pipeline_ingest = DataIngestionPipeline()
            pipeline_ingest.ingest_batch(DEFAULT_TEAM_URLS)

        self.vector_store = SimplePersistentVectorStore.load(self.store_path)
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

        context_docs = self.vector_store.search(question, top_k=8)
        if self._is_tournament_question(question):
            tournament_docs = self.vector_store.search(self._build_tournament_query(question), top_k=12)
            extracted = self._extract_latest_tournament(tournament_docs or context_docs)
            if extracted:
                return extracted
            return (
                "No encontré evidencia suficiente en el contexto para identificar el último torneo con fiabilidad. "
                "Indica el equipo (por ejemplo, 'último torneo de G2') y te respondo con precisión."
            )

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