"""
Aplicación del Chatbot RAG para responder preguntas sobre esports.

Este programa no hace scraping al arrancar. Solo carga el vector store
ya generado por el programa de ingesta y abre el chatbot.
"""

import json
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para importar rag/
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingestion_pipeline import RAGEngine


class EsportsChatbot:
    """Chatbot conversacional sobre esports."""

    def __init__(self):
        self.rag_engine = RAGEngine()
        self.conversation_history = []
        self.is_initialized = False

    def load_persisted_data(self):
        """Carga el vector store ya generado por el programa de ingesta."""
        print("\n" + "=" * 70)
        print("CARGANDO DATOS YA SCRAPEADOS")
        print("=" * 70)

        total_docs = self.rag_engine.vector_store.count()
        if total_docs > 0:
            self.is_initialized = True
            print(f"✓ {total_docs} documentos cargados desde disco\n")
            return True

        print("⚠️  No hay datos guardados en el vector store.")
        print("   Primero ejecuta: python chatbot/ingest.py")
        print("   El chatbot no se abrirá hasta que exista contenido guardado.\n")
        self.is_initialized = False
        return False
    
    def chat(self, user_input: str) -> str:
        """
        Procesa entrada del usuario y retorna respuesta.
        
        Args:
            user_input: Pregunta del usuario
            
        Returns:
            Respuesta del chatbot
        """
        if not self.is_initialized:
            return "El chatbot no está inicializado. Primero ejecuta el programa de ingesta."
        
        # Agregar a historial
        self.conversation_history.append({
            'user': user_input,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
        # Generar respuesta usando RAG
        result = self.rag_engine.answer_question(user_input)
        
        # Agregar respuesta a historial
        self.conversation_history[-1]['bot'] = result['response']
        
        return result['response']
    
    def run_interactive(self):
        """Inicia el chatbot en modo conversacional interactivo."""
        print("\n" + "=" * 70)
        print("CHATBOT RAG - ESPORTS")
        print("=" * 70)
        print("\nEscribe 'salir', 'exit' o 'quit' para terminar.\n")
        
        while True:
            try:
                user_input = input("Tú: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("\n✅ ¡Hasta luego!\n")
                    break
                
                print("\n🔄 Procesando...\n")
                response = self.chat(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n✅ ¡Hasta luego!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def save_conversation(self, filename: str = "conversation.json"):
        """Guarda el historial de conversación"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"✓ Conversación guardada en {filename}")


def main():
    """Función principal: abre el chatbot con datos ya scrapeados."""
    try:
        chatbot = EsportsChatbot()
        if chatbot.load_persisted_data():
            chatbot.run_interactive()
    except KeyboardInterrupt:
        print("\n\nChatbot cerrado\n")
    except Exception as e:
        print(f"\nError fatal: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
