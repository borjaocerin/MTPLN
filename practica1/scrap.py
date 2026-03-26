import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.mediamarkt.es/es/product/_apple-airpods-pro-3-2025-3a-gen-inalambricos-cancelacion-de-ruido-medicion-frecuencia-cardiaca-live-translation-chip-h2-usb-c-blanco-1606182.html"

try:
    driver.get(url)
    time.sleep(4) # Espera carga inicial

    # Aceptar cookies
    try:
        driver.find_element(By.ID, "pwa-consent-layer-accept-all-button").click()
    except: pass

    print("⬇️ Bajando hasta el fondo para cargar todas las reseñas...")
    # Hacemos un scroll más agresivo para forzar el renderizado
    for _ in range(30): 
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(1.5)

    # ESTRATEGIA: Buscamos bloques de reseña individuales
    # La clase sc-e8a38d00-0 que me pasaste es el contenedor de CADA reseña.
    bloques = driver.find_elements(By.CLASS_NAME, "sc-e8a38d00-0")
    
    lista_final = []

    for bloque in bloques:
        # Extraemos todos los textos con la clase que encontraste DENTRO del bloque
        textos_internos = bloque.find_elements(By.CLASS_NAME, "sc-59b6826e-0")
        
        # Unimos los textos del bloque para tener la reseña completa (Pros + Contras + Comentario)
        texto_completo = " | ".join([t.text for t in textos_internos if len(t.text) > 2])
        
        if texto_completo:
            lista_final.append(texto_completo)

    # Guardar resultados
    if lista_final:
        df = pd.DataFrame(lista_final, columns=["Reseña Completa"])
        df.to_csv("airpods_resenas_detalladas.csv", index=False, encoding="utf-8-sig")
        print(f"✅ ¡Hecho! Guardadas {len(lista_final)} reseñas con todo el detalle.")
        print(f"Muestra: {lista_final[0][:150]}...")
    else:
        print("❌ No se pudieron capturar los bloques. Prueba a aumentar el scroll.")

finally:
    driver.quit()