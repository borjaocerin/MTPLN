# Lanzar el Proyecto

## 1. Activar entorno virtual
```powershell
& .venv\Scripts\Activate.ps1
```

## 2. Instalar dependencias (si es necesario)
```bash
pip install -r requirements.txt
```

## 3. Opción A: Ingesta de equipos (30 por defecto)
```bash
python chatbot/ingest.py --sources-file sources_example.json --max-teams 30
```

## 4. Opción B: Pipeline completo
```bash
python run_full_pipeline.py
```

## 5. Ejecutar EDA (obligatorio)
```bash
jupyter nbconvert --to notebook --execute EDA_Analisis.ipynb
```

## 6. Iniciar chatbot
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
Chatbot (consultas)
```
