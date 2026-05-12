#!/usr/bin/env python3
"""Script principal para ejecutar la ingesta de equipos y abrir el chatbot."""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))


def run_quick_ingest():
    """Ingesta con configuración rápida por defecto."""
    print("\n" + "=" * 70)
    print("INGESTA RÁPIDA: Equipos por Defecto")
    print("=" * 70)
    from chatbot.ingest import DataIngestionPipeline, DEFAULT_TEAM_URLS

    pipeline = DataIngestionPipeline()
    try:
        print(f"\nTeams: {len(DEFAULT_TEAM_URLS)}")
        docs = pipeline.ingest_batch(DEFAULT_TEAM_URLS)
        print(f"\n✅ Documentos ingestados: {len(docs)}")
    finally:
        pipeline.cleanup()


def run_massive_ingest(sources_file):
    """Ingesta masiva desde archivo JSON."""
    print("\n" + "=" * 70)
    print("INGESTA MASIVA: Desde archivo JSON externo")
    print("=" * 70)
    from chatbot.ingest import (
        DataIngestionPipeline,
        load_sources_file,
    )

    try:
        team_urls = load_sources_file(sources_file)
    except Exception as e:
        print(f"❌ Error leyendo {sources_file}: {e}")
        return

    pipeline = DataIngestionPipeline()
    try:
        print(f"\nTeams: {len(team_urls)}")
        docs = pipeline.ingest_batch(team_urls)
        print(f"\n✅ Documentos ingestados: {len(docs)}")
    finally:
        pipeline.cleanup()


def run_limited_ingest(max_teams):
    """Ingesta limitada (útil para pruebas)."""
    print("\n" + "=" * 70)
    print("INGESTA LIMITADA: Primeros N equipos")
    print("=" * 70)
    from chatbot.ingest import (
        DataIngestionPipeline,
        DEFAULT_TEAM_URLS,
    )

    pipeline = DataIngestionPipeline()
    try:
        teams = DEFAULT_TEAM_URLS[:max_teams] if max_teams else DEFAULT_TEAM_URLS
        print(f"\nTeams: {len(teams)}")
        docs = pipeline.ingest_batch(teams)
        print(f"\n✅ Documentos ingestados: {len(docs)}")
    finally:
        pipeline.cleanup()


def run_custom_ingest(teams):
    """Ingesta con URLs personalizadas."""
    print("\n" + "=" * 70)
    print("INGESTA PERSONALIZADA: URLs de equipos específicas")
    print("=" * 70)
    from chatbot.ingest import DataIngestionPipeline

    pipeline = DataIngestionPipeline()
    try:
        print(f"\nTeams: {len(teams)}")
        docs = pipeline.ingest_batch(teams)
        print(f"\n✅ Documentos ingestados: {len(docs)}")
    finally:
        pipeline.cleanup()


def run_chatbot():
    """Inicia el chatbot conversacional."""
    from chatbot.app import EsportsChatbot

    chatbot = EsportsChatbot()
    if chatbot.load_persisted_data():
        chatbot.run_interactive()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "quick":
        run_quick_ingest()
        print("\n🎮 Abriendo chatbot...")
        run_chatbot()

    elif mode == "massive":
        sources_file = None
        if "--sources-file" in sys.argv:
            idx = sys.argv.index("--sources-file")
            if idx + 1 < len(sys.argv):
                sources_file = sys.argv[idx + 1]

        if not sources_file:
            print("❌ Requiere --sources-file <ruta>")
            sys.exit(1)

        run_massive_ingest(sources_file)
        print("\n🎮 Abriendo chatbot...")
        run_chatbot()

    elif mode == "limited":
        max_teams = None

        if "--max-teams" in sys.argv:
            idx = sys.argv.index("--max-teams")
            if idx + 1 < len(sys.argv):
                max_teams = int(sys.argv[idx + 1])

        run_limited_ingest(max_teams)
        print("\n🎮 Abriendo chatbot...")
        run_chatbot()

    elif mode == "custom":
        teams = []

        if "--teams" in sys.argv:
            idx = sys.argv.index("--teams")
            while idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
                teams.append(sys.argv[idx + 1])
                idx += 1

        if not teams:
            print("❌ Requiere --teams")
            sys.exit(1)

        run_custom_ingest(teams)
        print("\n🎮 Abriendo chatbot...")
        run_chatbot()

    elif mode == "chat":
        print("\n🎮 Abriendo chatbot (sin reingestar)...")
        run_chatbot()

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
