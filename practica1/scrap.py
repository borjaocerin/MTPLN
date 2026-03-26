import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURACIÓN ---
urls_productos = [
    "https://www.mediamarkt.es/es/product/_apple-airpods-pro-3-2025-3a-gen-inalambricos-cancelacion-de-ruido-medicion-frecuencia-cardiaca-live-translation-chip-h2-usb-c-blanco-1606182.html",
    # Añade aquí todas las URLs que quieras:
    # "https://www.mediamarkt.es/es/product/_ejemplo-producto-2.html",
    # "https://www.mediamarkt.es/es/product/_ejemplo-producto-3.html",
]

options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

todas_las_resenas = [] # Lista para acumular TODO

try:
    for index, url in enumerate(urls_productos):
        print(f"\n📦 Procesando producto {index + 1}/{len(urls_productos)}: {url[:60]}...")
        driver.get(url)
        time.sleep(5) 

        # Aceptar cookies (solo suele pedirlo en la primera carga, pero por si acaso)
        try:
            driver.find_element(By.ID, "pwa-consent-layer-accept-all-button").click()
            print("✅ Cookies aceptadas.")
        except: pass

        # Scroll dinámico
        print("⬇️ Bajando para cargar reseñas...")
        for i in range(25): 
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)

        # Extracción de bloques
        bloques = driver.find_elements(By.CLASS_NAME, "sc-e8a38d00-0")
        print(f"🔎 Bloques encontrados en este producto: {len(bloques)}")

        for bloque in bloques:
            # Intentar desplegar "Mostrar más" si existe
            try:
                btn_mas = bloque.find_element(By.CSS_SELECTOR, "button[data-test='expand-button']")
                driver.execute_script("arguments[0].click();", btn_mas)
            except: pass

            textos_internos = bloque.find_elements(By.CLASS_NAME, "sc-59b6826e-0")
            texto_completo = " | ".join([t.text for t in textos_internos if len(t.text) > 2])
            
            if texto_completo:
                # Guardamos el texto y la URL para saber a qué producto pertenece
                todas_las_resenas.append({
                    "Producto_URL": url,
                    "Reseña": texto_completo
                })

    # --- GUARDAR RESULTADOS FINALES ---
    if todas_las_resenas:
        df = pd.DataFrame(todas_las_resenas)
        df.to_csv("scraping_mediamarkt_multiple.csv", index=False, encoding="utf-8-sig")
        print(f"\n✨ ¡FINALIZADO! Se han guardado {len(todas_las_resenas)} reseñas totales en 'scraping_mediamarkt_multiple.csv'.")
    else:
        print("\n❌ No se extrajo ninguna reseña de la lista de productos.")

finally:
    driver.quit()