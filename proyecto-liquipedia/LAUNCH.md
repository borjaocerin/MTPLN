# Lanzar el Proyecto

## 1. Activar entorno virtual
```powershell
& .venv\Scripts\Activate.ps1
```

## 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 3. Opcional: elegir otro modelo local
La primera ejecución del chatbot descargará el modelo local Phi-3-mini desde Hugging Face. Si quieres usar otro modelo local, define `LOCAL_LLM_MODEL` antes de arrancar.

## 4. Ingesta de equipos (30 por defecto)
```bash
python chatbot/ingest.py --sources-file sources_example.json --max-teams 30
```

**Alternativa con run_full_pipeline.py:**
```bash
python run_full_pipeline.py limited --max-teams 30
```

## 5. Iniciar chatbot
```bash
python chatbot/app.py
```

Si `python` no está disponible en tu terminal de Windows, usa:
```bash
py -3 chatbot/ingest.py --sources-file sources_example.json --max-teams 30
py -3 chatbot/app.py
```

## 6. EDA
El notebook de EDA ya no es necesario para arrancar el chatbot. Si quieres ejecutar el análisis, puedes hacerlo aparte con:
```bash
jupyter nbconvert --to notebook --execute EDA_Analisis.ipynb
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
