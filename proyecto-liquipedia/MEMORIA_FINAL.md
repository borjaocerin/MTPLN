# MEMORIA FINAL: PROYECTO TEXT MINING & NLP
## Análisis de Equipos de Counter-Strike desde Liquipedia

**Autor**: Proyecto Líquipedia NLP  
**Fecha**: 2026-05-12  
**Lenguaje**: Español  
**Institución**: MTPLN  

---

## 1. OBJETIVO DEL PROYECTO

Desarrollar un sistema de **minería de textos y procesamiento de lenguaje natural (NLP)** basado en datos de equipos profesionales de Counter-Strike extraídos desde Liquipedia, con traducción al español y un chatbot RAG (Retrieval-Augmented Generation) capaz de responder preguntas sobre estos equipos en español.

### Alcance
- ✅ Web scraping de equipos desde Liquipedia
- ✅ Limpieza y normalización de textos
- ✅ Traducción al español (deep-translator)
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Sistema RAG con búsqueda BM25
- ✅ Chatbot interactivo en español
- ✅ Análisis de sesgos
- ✅ Documentación actualizada

---

## 2. METODOLOGÍA

### 2.1 Recolección de Datos

**Fuente**: Liquipedia (https://liquipedia.net/counterstrike/)

**Herramientas**:
- **Selenium WebDriver**: Navegación automatizada con scroll para capturar contenido dinámico
- **BeautifulSoup4**: Parsing HTML y extracción de información estructurada
- **Estrategia**: Scroll progresivo y creación/limpieza del driver por URL para evitar fugas de memoria

**Equipos recolectados**: 30 (ver `data/latest_ingest.json`)

**Datos extraídos por equipo**:
- Nombre del equipo, URL, infobox (ubicación, fundación, staff), texto completo, referencias, timestamp de extracción

### 2.2 Limpieza y Preprocesamiento

**Pasos**:
1. Eliminación de HTML y artefactos de navegación
2. Normalización de espacios y encodings
3. Remoción de tokens de paginación ('saltar')
4. Traducción por chunks (<=1000 chars) con fallback
5. Validación de calidad (longitud mínima, ausencia de errores)

### 2.3 Ingesta en Vector Store

**Componente**: `rag/ingestion_pipeline.py` (SimplePersistentVectorStore + BM25)

Proceso: limpieza → traducción → conversión a documentos → ingestión en `chatbot/data/vector_store/store.json`.

---

## 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)

### 3.1 Estadísticas del Corpus

| Métrica | Valor |
|---------|-------|
| **Total de documentos** | 30 |
| **Total de caracteres** | 794,467 |
| **Total de palabras** | 138,316 |
| **Promedio de palabras/doc** | 4,610.53 |
| **Promedio de caracteres/doc** | 26,482.23 |
| **Máximo de palabras (por doc)** | 9,503 |
| **Mínimo de palabras (por doc)** | 41 |
| **Desv. Estándar (palabras)** | 2,344.97 |

### 3.2 Análisis de Vocabulario

| Métrica | Valor |
|---------|-------|
| **Total de tokens (conteo)** | 81,685 |
| **Vocabulario único (types)** | 8,490 |
| **Type-Token Ratio (TTR)** | 10.39% |
| **Palabra más frecuente** | "saltar" (alta incidencia por artefactos de paginación) |

Observación: el vocabulario es amplio pero contiene ruido (metadatos y tokens de navegación); limpieza adicional reduce este ruido.

### 3.3 Validación de Calidad

| Aspecto | Resultado |
|---------|-----------|
| **Documentos sin nombre** | 0/30 |
| **Documentos sin URL** | 0/30 |
| **Documentos vacíos (<100 chars)** | 0/30 |
| **Documentos muy cortos (<500 palabras)** | 2/30 |
| **Documentos muy largos (>5000 palabras)** | 13/30 |
| **Documentos duplicados** | 0/30 |

Calidad general: mayormente buena. La principal debilidad es la alta variabilidad en longitud entre documentos.

### 3.4 Visualizaciones

Las figuras principales se guardaron en `EDA_Visualizaciones.png` (distribución de longitudes, top tokens, TTR por doc, y cobertura por país detectado).

---

## 4. ANÁLISIS DE SESGOS (RESULTADOS)

Se detectaron sesgos relevantes tras analizar cobertura, geografía, volumen y calidad de extracción.

### 4.1 Sesgo de Fuente y Cobertura
- Todo el corpus proviene de Liquipedia → sesgo de origen (estructura editorial homogénea).
- Impacto: MEDIO-ALTO (puede faltar contenido local o enfoques periodísticos alternativos).

### 4.2 Sesgo Geográfico
- Conteo automático (resumen): Brasil (4), Estados Unidos (3), Alemania (2), Ucrania (2), Unknown/no detectado (2), resto disperso.
- Observación: etiquetas del infobox presentan formatos inconsistentes (acentos/encodings), por lo que algunas ubicaciones requieren normalización.

### 4.3 Sesgo de Cantidad / Prominencia
- Heterogeneidad en longitud: ~43% documentos muy largos, ~7% muy cortos.
- Impacto: ALTO — los equipos con páginas extensas dominan recuperación lexical y métricas de frecuencia.

### 4.4 Sesgo Temporal y de Contenido
- Frecuencia alta de tokens año/cronología y tokens de navegación → indica predominio de cronología y metadatos en el texto.

### 4.5 Sesgo de Calidad de Extracción
- Hay 2 documentos con país no detectado y varios con cadenas parciales (p. ej. 'Brasil Regi?n') por issues de encoding.

### Recomendaciones resumidas
1. Expandir corpus con equipos regionales (Asia, Oceanía) y equipos emergentes.  
2. Normalizar metadatos del infobox (mapping de países, normalización de encodings).  
3. Limpiar tokens de navegación y fechas redundantes antes del indexado.  
4. Aplicar chunking y/o ponderación para compensar documentos muy largos.  
5. Recalcular EDA tras las correcciones y documentar la versión del corpus.

---

## 5. ANEXOS

- Archivo de ingestión persistente: data/latest_ingest.json (30 documentos)  
- Vector store: chatbot/data/vector_store/store.json (BM25 index, 30 documentos)  
- Visualizaciones EDA: EDA_Visualizaciones.png

## 6. CONCLUSIONES Y SIGUIENTES PASOS

1. Ingesta de 30 equipos completada y BM25 poblado.  
2. El corpus permite demos de chatbot RAG, pero necesita mitigaciones para reducir sesgos (ver recomendaciones).  
3. Siguientes acciones prioritarias: expandir con +30 equipos (Asia/Oceanía/emergentes), normalizar infobox y encodings, regenerar índice y volver a ejecutar EDA.

---

**Fin de la memoria (actualizada con EDA fecha 2026-05-12).**
