import requests
from bs4 import BeautifulSoup
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from deep_translator import GoogleTranslator

class LiquipediaTransfersScraper:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 15) # Más tiempo de espera
        self.translator = GoogleTranslator(source='auto', target='es')
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def get_soup(self, url):
        """Accede a la página y hace scroll para asegurar que carguen las tablas"""
        print(f"   [!] Abriendo navegador para: {url}")
        try:
            self.driver.get(url)
            # Hacer scroll hacia abajo para disparar la carga de elementos dinámicos
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(3)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5) 
            
            return BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            print(f"   [!] Error al cargar página: {e}")
            return None

    def extract_transfer_references(self, soup):
        """Busca referencias de forma masiva en el cuerpo de la página"""
        if not soup: return []
        refs = []
        
        # Intentamos localizar el contenedor principal de Liquipedia
        content_area = soup.find('div', {'id': 'mw-content-text'}) or soup.find('body')

        # Buscamos TODOS los enlaces externos (las famosas flechitas y números [1])
        links = content_area.find_all('a', class_='external text')
        
        # Filtros de dominios que sí son fuentes de noticias
        fuentes_validas = ["x.com", "twitter.com", "vlr.gg", "hltv.org", "facebook.com", "instagram.com", "escharts", "dexerto", "dust2"]

        for a in links:
            url = a.get('href', '')
            if url.startswith('http') and any(dom in url for dom in fuentes_validas):
                # Evitamos duplicados de la propia Liquipedia que a veces se marcan como externos
                if "liquipedia.net" not in url:
                    label = a.get_text(strip=True) or "Fuente"
                    refs.append({"label": label, "url": url})

        # Limpiar duplicados de la lista
        unique = {r['url']: r for r in refs}.values()
        return list(unique)

    def scrape_twitter(self, url):
        try:
            self.driver.get(url)
            # Esperamos específicamente al tuit
            element = self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@data-testid="tweetText"]')))
            text = element.text
            try:
                date = self.driver.find_element(By.XPATH, '//time').get_attribute("datetime")
            except: date = "N/A"
            return {"text": text, "text_es": self.traducir(text), "date": date}
        except: return None

    def scrape_generic(self, url):
        try:
            self.driver.get(url)
            time.sleep(4)
            paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
            texts = [p.text for p in paragraphs if len(p.text) > 50]
            if not texts: return None
            content = " ".join(texts[:5])
            return {"text": content, "text_es": self.traducir(content)}
        except: return None

    def scrape_external_content(self, url):
        if "twitter.com" in url or "x.com" in url:
            return self.scrape_twitter(url)
        return self.scrape_generic(url)

    def traducir(self, texto):
        if not texto: return None
        try: return self.translator.translate(texto)
        except: return texto

    def run(self, config):
        all_data = []
        processed_urls = set()
        
        # Lista de meses en inglés como pide Liquipedia
        meses = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]

        for game, years in config.items():
            for year in years:
                for mes in meses:
                    # Construcción de la URL: /Portal:Transfers/2025/January
                    url = f"https://liquipedia.net/{game}/Player_Transfers/{year}/{mes}"
                    print(f"\n>> Analizando: {url}")
                    
                    soup = self.get_soup(url)
                    if not soup:
                        continue
                        
                    refs = self.extract_transfer_references(soup)
                    
                    if not refs:
                        print(f"   [!] No hay transferencias registradas para {mes} {year}.")
                        continue

                    print(f"   Encontradas {len(refs)} fuentes reales en {mes}.")

                    for ref in refs:
                        if ref['url'] in processed_urls: 
                            continue
                            
                        print(f"   - Scrapeando: {ref['url']}")
                        content = self.scrape_external_content(ref['url'])
                        
                        if content:
                            all_data.append({
                                "game": game, 
                                "year": year, 
                                "month": mes, # Añadimos el mes al JSON resultante
                                "url": ref['url'],
                                "content": content.get("text"),
                                "content_es": content.get("text_es"),
                                "date": content.get("date")
                            })
                            processed_urls.add(ref['url'])
                        time.sleep(1) # Respeto entre peticiones de fuentes externas
                        
        return all_data

if __name__ == "__main__":
    # Configuración de búsqueda
    search_config = {
        "valorant": [2024, 2025],
        "counterstrike": [2024, 2025]
    }

    scraper = LiquipediaTransfersScraper()
    try:
        data = scraper.run(search_config)
        # Guardamos con un nombre descriptivo
        with open("fichajes_mensuales.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n¡Éxito! {len(data)} registros mensuales guardados.")
    finally:
        scraper.driver.quit()
