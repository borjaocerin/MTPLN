import time
import re # Librería para limpieza de texto
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURACIÓN ---
urls_productos = [
    "https://www.mediamarkt.es/es/product/_apple-airpods-pro-3-2025-3a-gen-inalambricos-cancelacion-de-ruido-medicion-frecuencia-cardiaca-live-translation-chip-h2-usb-c-blanco-1606182.html",
    "https://www.mediamarkt.es/es/product/_auriculares-true-wireless-huawei-freeclip-2-38h-autonomia-ip57-open-ear-llamadas-nitidas-conexion-a-dos-dispositivos-azul-1617474.html",
    "https://www.mediamarkt.es/es/product/_auriculares-true-wireless-xiaomi-buds-8-lite-36h-autonomia-cancelacion-de-ruido-ip54-diseno-ligero-negro-1621472.html",
    "https://www.mediamarkt.es/es/product/_auriculares-inalambricos-comodidad-cancelacion-de-ruido-y-resistencia-al-agua-kinglucky-intraurales-rosa-161895991.html",
    "https://www.mediamarkt.es/es/product/_auriculares-true-wireless-redmi-buds-6-play-xiaomi-intraurales-negro-145405399.html",
    "https://www.mediamarkt.es/es/product/_auriculares-true-wireless-samsung-galaxy-buds3-pro-anc-bluetooth-54-audio-24bits-3-microfonos-reproduccion-hasta-30h-dual-amp-grafito-1578034.html"
]
options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

solo_textos = [] 

def limpiar_texto(texto):
    # 1. Eliminar saltos de línea y tabulaciones
    texto = texto.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 2. Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto)
    # 3. Quitar espacios al inicio y final
    return texto.strip()

try:
    for index, url in enumerate(urls_productos):
        print(f"\n Procesando producto {index + 1}/{len(urls_productos)}")
        driver.get(url)
        time.sleep(5) 

        try:
            driver.find_element(By.ID, "pwa-consent-layer-accept-all-button").click()
        except: pass

        for i in range(15): 
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(0.5)

        bloques = driver.find_elements(By.CLASS_NAME, "sc-e8a38d00-0")
        
        for bloque in bloques:
            try:
                btn_mas = bloque.find_element(By.CSS_SELECTOR, "button[data-test='expand-button']")
                driver.execute_script("arguments[0].click();", btn_mas)
                time.sleep(0.2)
            except: pass

            textos_internos = bloque.find_elements(By.CLASS_NAME, "sc-59b6826e-0")
            # Unimos todo el contenido de la reseña en una sola cadena
            texto_bruto = " ".join([t.text for t in textos_internos if len(t.text) > 1])
            
            # Limpiamos el texto para que no rompa el CSV
            texto_limpio = limpiar_texto(texto_bruto)
            
            if texto_limpio:
                solo_textos.append(texto_limpio)

    # --- GUARDAR RESULTADOS ---
    if solo_textos:
        # Convertimos a DataFrame y guardamos
        df = pd.DataFrame(solo_textos, columns=["Reseña"])
        # Usamos quoting=1 (QUOTE_ALL) para asegurar que el texto vaya entre comillas
        df.to_csv("comentarios_limpios.csv", index=False, encoding="utf-8-sig", quoting=1)
        print(f"\n Se guardaron {len(solo_textos)} reseñas en 'comentarios_limpios.csv'.")
    else:
        print("\n No se encontraron reseñas.")

finally:
    driver.quit()