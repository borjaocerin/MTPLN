import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from chatbot.ingest import DataIngestionPipeline, DEFAULT_STORE_FILE, DEFAULT_TEAM_URLS, SimplePersistentVectorStore
else:
    from .ingest import DataIngestionPipeline, DEFAULT_STORE_FILE, DEFAULT_TEAM_URLS, SimplePersistentVectorStore


class EsportsChatbot:
    def __init__(self, model_name: str | None = None, store_path: str | Path | None = None):
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_FILE
        self.vector_store = SimplePersistentVectorStore.load(self.store_path)
        self.generator = self._load_generator()

    def _load_generator(self):
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_type = rope_scaling.get("rope_type")
                if rope_type == "default":
                    config.rope_scaling = None
                elif "type" not in rope_scaling and rope_type:
                    config.rope_scaling = dict(rope_scaling)
                    config.rope_scaling["type"] = rope_type

            config._attn_implementation = "eager"

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model_kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                model_kwargs["dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["dtype"] = torch.float32

            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["config"] = config

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            model.config.use_cache = False
            device = 0 if torch.cuda.is_available() else -1
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
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
            context_lines.append(f"[{index}] {doc['name']} ({doc['url']}): {doc['text'][:900]}")

        context = "\n\n".join(context_lines) if context_lines else "No hay contexto disponible."
        return (
            "Eres un asistente en español sobre equipos de Counter-Strike y su historial en Liquipedia. "
            "Responde solo con información apoyada en el contexto. Si no hay suficiente evidencia, dilo claramente.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n"
            "Respuesta:"
        )

    def answer(self, question: str) -> str:
        if self.generator is None:
            raise RuntimeError("El modelo local no está disponible; no se puede responder sin LLM.")

        context_docs = self.vector_store.search(question, top_k=4)
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
                max_new_tokens=96,
                return_full_text=False,
                pad_token_id=getattr(self.generator.tokenizer, "eos_token_id", None),
                use_cache=False,
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
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct"))
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