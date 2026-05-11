"""
Pipeline de ingesta de datos para el chatbot RAG
Extrae datos de Liquipedia y los ingesta en el vector store
"""

import json
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para importar scraper/ y rag/
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.main_scraper import LiquipediaTransfersScraper
from scraper.cleaner import TextCleaner, validate_text_quality
from rag.ingestion_pipeline import RAGEngine

# Intentar usar deep-translator para traducir al español. Si no está disponible,
# usamos un fallback que devuelve el texto original.
try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except Exception:
    GoogleTranslator = None
    _HAS_TRANSLATOR = False


DEFAULT_TEAM_URLS = [
    # Counter-Strike
    "https://liquipedia.net/counterstrike/G2_Esports",
    "https://liquipedia.net/counterstrike/FaZe_Clan",
    "https://liquipedia.net/counterstrike/Natus_Vincere",
    "https://liquipedia.net/counterstrike/Team_Vitality",
    "https://liquipedia.net/counterstrike/MOUZ",
    "https://liquipedia.net/counterstrike/Team_Spirit",
    "https://liquipedia.net/counterstrike/Complexity_Gaming",
    "https://liquipedia.net/counterstrike/FNATIC",
    "https://liquipedia.net/counterstrike/Cloud9",
    "https://liquipedia.net/counterstrike/HEROIC",
    # VALORANT
    "https://liquipedia.net/valorant/Fnatic",
    "https://liquipedia.net/valorant/Sentinels",
    "https://liquipedia.net/valorant/Paper_Rex",
    "https://liquipedia.net/valorant/Team_Heretics",
    "https://liquipedia.net/valorant/Gen.G_Esports",
    "https://liquipedia.net/valorant/DRX",
    "https://liquipedia.net/valorant/LOUD",
    "https://liquipedia.net/valorant/Leviat%C3%A1n",
    "https://liquipedia.net/valorant/100_Thieves",
    "https://liquipedia.net/valorant/FUT_Esports",
]

DEFAULT_PLAYER_URLS = [
    # Counter-Strike
    "https://liquipedia.net/counterstrike/NiKo",
    "https://liquipedia.net/counterstrike/m0NESY",
    "https://liquipedia.net/counterstrike/ZywOo",
    "https://liquipedia.net/counterstrike/donk",
    "https://liquipedia.net/counterstrike/s1mple",
    "https://liquipedia.net/counterstrike/device",
    # VALORANT
    "https://liquipedia.net/valorant/Boaster",
    "https://liquipedia.net/valorant/Derke",
    "https://liquipedia.net/valorant/tenz",
    "https://liquipedia.net/valorant/Zekken",
    "https://liquipedia.net/valorant/Mako",
    "https://liquipedia.net/valorant/aspas",
]


def dedupe_urls(urls: list) -> list:
    """Deduplica URLs manteniendo orden de aparición."""
    seen = set()
    result = []
    for url in urls:
        if not url or not isinstance(url, str):
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result


def load_sources_file(file_path: str) -> tuple[list, list]:
    """Carga equipos y jugadores desde un JSON externo.

    Formato esperado:
    {
      "teams": ["https://...", "https://..."],
      "players": ["https://...", "https://..."]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    teams = payload.get("teams", [])
    players = payload.get("players", [])
    return dedupe_urls(teams), dedupe_urls(players)


class DataIngestionPipeline:
    """Pipeline para ingestar datos de Liquipedia en el chatbot"""
    
    def __init__(self):
        self.scraper = LiquipediaTransfersScraper()
        self.cleaner = TextCleaner()
        self.rag_engine = RAGEngine()
        self.ingested_docs = []
        # Configuración de traducción
        self.translator_available = _HAS_TRANSLATOR
        if self.translator_available:
            try:
                # Inicializar traductor a español
                self.translator = GoogleTranslator(source='auto', target='es')
            except Exception:
                self.translator_available = False
                self.translator = None
        else:
            self.translator = None

    def translate_text(self, text: str) -> str:
        """
        Traduce `text` al español si el traductor está disponible.
        Si falla, devuelve el texto original.
        """
        if not text:
            return text
        if not self.translator_available or self.translator is None:
            return text
        try:
            # Dividir por párrafos y luego en trozos de tamaño limitado
            # para forzar traducción cuando el texto viene en una sola línea.
            parts = [p for p in text.split('\n\n') if p.strip()]
            if not parts:
                parts = [text]

            max_chunk = 1000
            translated_parts = []
            for p in parts:
                if len(p) <= max_chunk:
                    try:
                        t = self.translator.translate(p)
                    except Exception:
                        t = p
                    translated_parts.append(t)
                    continue

                # dividir en trozos más pequeños
                chunks = [p[i:i+max_chunk] for i in range(0, len(p), max_chunk)]
                translated_chunks = []
                for c in chunks:
                    try:
                        tc = self.translator.translate(c)
                    except Exception:
                        tc = c
                    translated_chunks.append(tc)
                translated_parts.append(''.join(translated_chunks))

            return '\n\n'.join(translated_parts)
        except Exception:
            return text
    
    def extract_team_data(self, url: str) -> dict:
        """
        Extrae datos de equipo de Liquipedia.
        
        Args:
            url: URL del equipo en Liquipedia
            
        Returns:
            Datos del equipo p
        """
        print(f"Extrayendo equipo: {url}")
        team_data = self.scraper.scrape_team_page(url)
        
        if not team_data:
            print(f"  [!] Error al extraer {url}")
            return None
        
        print(f"  [OK] {team_data.get('name')}")
        return team_data
    
    def extract_player_data(self, url: str) -> dict:
        """
        Extrae datos de jugador de Liquipedia.
        
        Args:
            url: URL del jugador en Liquipedia
            
        Returns:
            Datos del jugador
        """
        print(f"Extrayendo jugador: {url}")
        player_data = self.scraper.scrape_player_page(url)
        
        if not player_data:
            print(f"  [!] Error al extraer {url}")
            return None
        
        print(f"  [OK] {player_data.get('name')}")
        return player_data
    
    def convert_to_documents(self, data: dict, data_type: str) -> list:
        """
        Convierte datos extraídos a documentos para ingesta.
        
        Args:
            data: Datos extraídos (equipo o jugador)
            data_type: Tipo de dato ('team' o 'player')
            
        Returns:
            Lista de documentos
        """
        documents = []

        def add_document(text: str, metadata: dict):
            clean_text = self.cleaner.clean_text(text)
            # Traducir al español antes de validar/añadir
            esp_text = self.translate_text(clean_text)
            if validate_text_quality(esp_text, min_length=30):
                # Guardar texto traducido
                documents.append({
                    'text': esp_text,
                    'metadata': metadata
                })

        name = data.get('name', 'Unknown')

        # 1) Documento combinado principal con toda la información útil
        combined_parts = []

        if data.get('infobox'):
            infobox_text = " | ".join([f"{k}: {v}" for k, v in data['infobox'].items()])
            combined_parts.append(f"INFOBOX: {infobox_text}")

        if data.get('info'):
            info_text = " | ".join([f"{k}: {v}" for k, v in data['info'].items()])
            combined_parts.append(f"INFOBOX: {info_text}")

        for key, value in data.items():
            if not isinstance(value, str):
                continue
            if key in {'url', 'type', 'name', 'extracted_at'}:
                continue
            if key.endswith('_es'):
                continue
            if key in {'combined_text'}:
                continue

            label = key.replace('_', ' ').upper()
            combined_parts.append(f"{label}: {value}")

        if data.get('external_references'):
            ref_lines = []
            for ref in data['external_references'][:10]:
                if isinstance(ref, dict):
                    ref_lines.append(f"{ref.get('label', 'Fuente')}: {ref.get('url', '')}")
            if ref_lines:
                combined_parts.append("REFERENCIAS EXTERNAS: " + " | ".join(ref_lines))

        combined_text = "\n\n".join(combined_parts)
        add_document(
            combined_text,
            {
                'name': name,
                'type': data_type,
                'url': data.get('url'),
                'kind': 'combined',
                'sections': [k for k, v in data.items() if isinstance(v, str) and k not in {'url', 'type', 'name', 'extracted_at'}]
            }
        )

        # 2) Documento por cada sección textual útil para aumentar la cobertura
        for key, value in data.items():
            if not isinstance(value, str):
                continue
            if key in {'url', 'type', 'name', 'extracted_at'}:
                continue
            if key.endswith('_es'):
                continue
            if key == 'combined_text':
                continue

            add_document(
                f"{key.replace('_', ' ').title()}: {value}",
                {
                    'name': name,
                    'type': data_type,
                    'url': data.get('url'),
                    'kind': key
                }
            )

            # Para secciones largas, crear chunks adicionales para no perder detalle.
            if len(value) > 1400:
                chunk_size = 1200
                overlap = 180
                start = 0
                chunk_idx = 0
                while start < len(value):
                    end = min(len(value), start + chunk_size)
                    chunk = value[start:end]
                    add_document(
                        f"{key.replace('_', ' ').title()} (chunk {chunk_idx + 1}): {chunk}",
                        {
                            'name': name,
                            'type': data_type,
                            'url': data.get('url'),
                            'kind': f"{key}_chunk",
                            'chunk_index': chunk_idx,
                        }
                    )
                    if end >= len(value):
                        break
                    start = max(end - overlap, start + 1)
                    chunk_idx += 1

        return documents
    
    def ingest_batch(self, team_urls: list = None, player_urls: list = None):
        """
        Ingesta múltiples equipos y jugadores.
        
        Args:
            team_urls: Lista de URLs de equipos
            player_urls: Lista de URLs de jugadores
        """
        print("\n" + "="*60)
        print("PIPELINE DE INGESTA")
        print("="*60)
        
        all_documents = []
        
        # Extraer equipos
        if team_urls:
            print("\n[1/2] Extrayendo equipos...")
            for url in team_urls:
                try:
                    team_data = self.extract_team_data(url)
                    if team_data:
                        docs = self.convert_to_documents(team_data, 'team')
                        all_documents.extend(docs)
                        
                        import time
                        time.sleep(2)  # Respetar servidor
                except Exception as e:
                    print(f"  [ERR] Error: {str(e)[:50]}")
        
        # Extraer jugadores
        if player_urls:
            print("\n[2/2] Extrayendo jugadores...")
            for url in player_urls:
                try:
                    player_data = self.extract_player_data(url)
                    if player_data:
                        docs = self.convert_to_documents(player_data, 'player')
                        all_documents.extend(docs)
                        
                        import time
                        time.sleep(2)  # Respetar servidor
                except Exception as e:
                    print(f"  [ERR] Error: {str(e)[:50]}")
        
        # Ingestar en RAG engine
        if all_documents:
            print(f"\n[3/3] Ingesta en Vector Store...")
            print(f"  Documentos a ingestar: {len(all_documents)}")

            # Importante: limpiar el índice para evitar mezclar documentos legacy
            # con la nueva ingesta (causa respuestas fuera de contexto).
            self.rag_engine.vector_store.clear()
            self.rag_engine.ingest_documents(all_documents)
            self.ingested_docs = all_documents

            # Guardar un respaldo de los documentos generados para depuración
            try:
                backup_path = Path(__file__).parent.parent / 'data'
                backup_path.mkdir(exist_ok=True)
                with open(backup_path / 'latest_ingest.json', 'w', encoding='utf-8') as f:
                    import json
                    json.dump(all_documents, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        print("\n[COMPLETE] Pipeline completado")
        return all_documents
    
    def cleanup(self):
        """Limpia recursos"""
        if hasattr(self, 'scraper'):
            self.scraper.driver.quit()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de ingesta para Chatbot')
    parser.add_argument('--teams', '-t', type=str, nargs='+',
                       help='URLs de equipos a extraer')
    parser.add_argument('--players', '-p', type=str, nargs='+',
                       help='URLs de jugadores a extraer')
    parser.add_argument('--sources-file', type=str,
                       help='JSON con listas de teams/players para ingesta masiva')
    parser.add_argument('--max-teams', type=int, default=None,
                       help='Limita cuántos equipos se procesan')
    parser.add_argument('--max-players', type=int, default=None,
                       help='Limita cuántos jugadores se procesan')
    
    args = parser.parse_args()
    
    pipeline = DataIngestionPipeline()
    
    try:
        if args.sources_file:
            file_teams, file_players = load_sources_file(args.sources_file)
        else:
            file_teams, file_players = [], []

        team_urls = args.teams if args.teams else (file_teams if file_teams else DEFAULT_TEAM_URLS)
        player_urls = args.players if args.players else (file_players if file_players else DEFAULT_PLAYER_URLS)

        team_urls = dedupe_urls(team_urls)
        player_urls = dedupe_urls(player_urls)

        if args.max_teams is not None:
            team_urls = team_urls[: max(0, args.max_teams)]
        if args.max_players is not None:
            player_urls = player_urls[: max(0, args.max_players)]

        print(f"\nSe procesarán {len(team_urls)} equipos y {len(player_urls)} jugadores.")
        
        docs = pipeline.ingest_batch(team_urls, player_urls)
        print("\n✅ Scraping e ingesta completados.")
        print("   Ahora puedes abrir el chatbot con: python app.py")
        print(f"   Documentos guardados: {len(docs)}")
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
