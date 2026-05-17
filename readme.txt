
**1) Preparar entorno Python (recomendado: virtualenv)**

Windows (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r proyecto-liquipedia/requirements.txt
```

Unix / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r proyecto-liquipedia/requirements.txt
```

Notas:
- Si el proyecto requiere dependencias adicionales fuera de `proyecto-liquipedia/requirements.txt`, instálalas según se indique en mensajes de error.

**2) Ejecutar el pipeline principal**

El proyecto incluye un script de pipeline global. Ejecuta desde la raíz del repositorio:

```bash
python proyecto-liquipedia/run_full_pipeline.py
```

Este script orquesta las etapas principales (scraping, limpieza, ingestión y generación de artefactos). Revisa la salida por si faltan dependencias o claves.

**3) Ejecutar el scraper por separado**

Si solo quieres extraer datos antes de la ingestión:

```bash
python proyecto-liquipedia/scraper/main_scraper.py
```

Los datos limpios/resultados parciales se escriben en la carpeta `proyecto-liquipedia/data/`.

**4) Ingestión / RAG pipeline**

Para ejecutar solo la ingesta (vector store / RAG):

```bash
python proyecto-liquipedia/rag/ingestion_pipeline.py
```

Salida relevante:
- `proyecto-liquipedia/data/latest_ingest.json` — resumen de la última ingestión.
- `proyecto-liquipedia/chatbot/data/vector_store/store.json` — almacén vectorial usado por el chatbot.

**5) Levantar el chatbot (interfaz local)**

```bash
python proyecto-liquipedia/chatbot/app.py
```
