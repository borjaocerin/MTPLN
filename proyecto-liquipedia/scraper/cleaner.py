import json

def limpiar_datos(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Registros originales: {len(data)}")

    # 1. Palabras que indican que el contenido NO sirve (ruido)
    blacklist = [
        "cookie", "política de privacidad", "privacy policy", "utiliza cookies",
        "maliciosos", "verificar que no eres un bot", "cloudflare", "derechos reservados",
        "all rights reserved", "18 years of age", "newsletter", "suscríbete", 
        "configuración de cookies", "mantenimiento"
    ]

    # 2. Palabras que nos interesan (fichajes/cambios)
    keywords_fichajes = [
        "join", "welcome", "fichaje", "departure", "farewell", "roster", 
        "contract", "free agent", "lft", "bench", "replaces", "signing",
        "bienvenida", "despedida", "contrato", "agente libre"
    ]

    cleaned_data = []

    for entry in data:
        texto = entry.get("content")
        
        # Saltamos entradas vacías o muy cortas
        if not texto or len(texto) < 45:
            continue

        texto_low = texto.lower()

        # FILTRO A: Si tiene palabras de la lista negra, fuera
        if any(bad_word in texto_low for bad_word in blacklist):
            continue

        # FILTRO B: Priorizar fichajes
        # (Si quieres ser muy estricto, solo guarda si tiene una keyword de fichajes)
        es_fichaje = any(kw in texto_low for kw in keywords_fichajes)
        
        if es_fichaje:
            # Si pasa los filtros, lo añadimos
            cleaned_data.append(entry)

    # Guardar el resultado limpio
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"Registros después de la limpieza: {len(cleaned_data)}")
    print(f"Archivo guardado como: {output_file}")

if __name__ == "__main__":
    limpiar_datos('external_refs_multiple_es.json', 'fichajes_limpios.json')