# Lanzar el Proyecto

## 1. Activar entorno virtual
```powershell
& .venv\Scripts\Activate.ps1
```

## 2. Instalar dependencias (si es necesario)
```bash
pip install -r requirements.txt
```

La primera ejecución del chatbot descargará el modelo local Phi-3-mini desde Hugging Face. Si quieres usar otro modelo local, define `LOCAL_LLM_MODEL` antes de arrancar.

## 3. Ingesta de equipos (30 por defecto)
```bash
python chatbot/ingest.py --sources-file sources_example.json --max-teams 30
```

**Alternativa con run_full_pipeline.py:**
```bash
python run_full_pipeline.py limited --max-teams 30
```

## 4. Ejecutar EDA (obligatorio)
```bash
jupyter nbconvert --to notebook --execute EDA_Analisis.ipynb
```

## 5. Iniciar chatbot
```bash
python chatbot/app.py
```

---

**Archivos generados:**
- `data/latest_ingest.json` → documentos scrapeados
- `chatbot/data/vector_store/store.json` → índice BM25
- `EDA_Visualizaciones.png` → gráficos del análisis

**Arquitectura:**
```
Scraper (Selenium/BeautifulSoup)
    ↓
Cleaner (limpieza de texto)
    ↓
Translator (Google Translator)
    ↓
RAG Ingestion (SimplePersistentVectorStore + BM25)
    ↓
LLM local Phi-3-mini
    ↓
Chatbot (consultas)
```
