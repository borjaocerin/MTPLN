import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

class LiquipediaTeamScraper:
    def __init__(self):
        # No mantener driver persistente; crear uno nuevo para cada URL
        self.driver = None
        if GoogleTranslator is not None:
            try:
                self.translator = GoogleTranslator(source='auto', target='es')
            except Exception:
                self.translator = None
        else:
            self.translator = None
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def _create_driver(self):
        """Crea una nueva instancia de driver."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=options)

    def cleanup(self):
        """Cierra el navegador si sigue abierto."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

    def get_soup(self, url):
        """Accede a la página y hace scroll para asegurar que carguen las tablas"""
        print(f"   [!] Abriendo navegador para: {url}")
        try:
            # Crear driver nuevo para cada URL
            self.driver = self._create_driver()
            self.driver.set_page_load_timeout(20)
            
            self.driver.get(url)
            # Hacer scroll hacia abajo para disparar la carga de elementos dinámicos
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return soup
        except Exception as e:
            print(f"   [!] Error al cargar página: {e}")
            return None
        finally:
            # Limpiar driver inmediatamente después
            self.cleanup()

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
        if self.translator is None:
            return texto
        try: return self.translator.translate(texto)
        except: return texto

    def _extract_title(self, soup):
        heading = soup.find("h1", id="firstHeading")
        if heading and heading.get_text(strip=True):
            return heading.get_text(strip=True)
        title = soup.find("title")
        if title and title.get_text(strip=True):
            return title.get_text(strip=True).split("-")[0].strip()
        return "Unknown"

    def _extract_infobox(self, soup):
        infobox = {}
        table = soup.find("table", class_=lambda c: c and "infobox" in c)
        if not table:
            return infobox

        for row in table.find_all("tr"):
            key_el = row.find("th")
            val_el = row.find("td")
            if not key_el or not val_el:
                continue
            key = key_el.get_text(" ", strip=True)
            val = val_el.get_text(" ", strip=True)
            if key and val:
                infobox[key] = val
        return infobox

    def _extract_full_text(self, soup):
        root = soup.find("div", id="mw-content-text") or soup.find("body")
        if not root:
            return ""

        # Eliminamos elementos que normalmente añaden ruido.
        for bad in root.select("table.navbox, table.infobox, script, style"):
            bad.decompose()

        text = root.get_text(" ", strip=True)
        return " ".join(text.split())

    def _extract_first_paragraph(self, soup):
        if not soup:
            return ""

        container = soup.find(
            "div",
            class_=lambda c: c and "mw-content-ltr" in c and "mw-parser-output" in c,
            lang="en",
            dir="ltr",
        )
        if not container:
            container = soup.find(
                "div",
                class_=lambda c: c and "mw-content-ltr" in c and "mw-parser-output" in c,
            )
        if not container:
            return ""

        paragraph = container.find("p")
        if not paragraph:
            return ""

        text = paragraph.get_text(" ", strip=True)
        text = self.traducir(text)
        return " ".join(text.split())

    def _normalize_movement_year(self, text: str, year: str | None) -> str:
        if not year or year in text:
            return text

        text = text.strip()

        english_match = re.match(r"^([A-Za-z]+ \d{1,2}(?:st|nd|rd|th)?)(.*)$", text)
        if english_match:
            date_part = english_match.group(1)
            rest = english_match.group(2)
            return f"{date_part} {year}{rest}"

        spanish_match = re.match(r"^(\d{1,2}(?:st|nd|rd|th)? de [A-Za-záéíóúñü]+)(.*)$", text, flags=re.IGNORECASE)
        if spanish_match:
            date_part = spanish_match.group(1)
            rest = spanish_match.group(2)
            return f"{date_part} {year}{rest}"

        return f"{text} ({year})"

    def _extract_tournaments(self, soup):
        if not soup:
            return []

        tournaments_section = soup.find("div", class_="tournaments-list-type-list")
        if not tournaments_section:
            return []

        tournaments = []
        for item in tournaments_section.find_all("div", class_=lambda c: c and "tournaments-list-item" in c):
            name_div = item.find("div", class_="tournaments-list-item__name")
            if not name_div:
                continue

            link = name_div.find("a")
            if link and link.get_text(strip=True):
                tournaments.append(link.get_text(strip=True))
                continue

            text = name_div.get_text(" ", strip=True)
            if text:
                tournaments.append(text)

        return tournaments

    def extract_movements(self, soup):
        """Extrae los movimientos del equipo cuando 'Show All' está activo."""
        show_all = soup.find('li', class_='show-all active')
        if not show_all:
            return []

        tabs_content = soup.find('div', class_='tabs-content')
        if not tabs_content:
            return []

        movements = []
        for section in tabs_content.find_all('div', class_=lambda c: c and c.startswith('content')):
            year = None
            header = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if header:
                span = header.find('span')
                if span and span.get_text(strip=True).isdigit():
                    year = span.get_text(strip=True)
                else:
                    header_text = header.get_text(" ", strip=True)
                    if header_text.isdigit():
                        year = header_text

            for li in section.find_all('li'):
                text = li.get_text(separator=' ', strip=True)
                text = re.sub(r'\[\d+\]', '', text)
                text = ' '.join(text.split())
                text = self._normalize_movement_year(text, year)
                text = self.traducir(text)
                movements.append(text)

        if not movements:
            for ul in tabs_content.find_all('ul'):
                for li in ul.find_all('li'):
                    text = li.get_text(separator=' ', strip=True)
                    text = re.sub(r'\[\d+\]', '', text)
                    text = ' '.join(text.split())
                    text = self.traducir(text)
                    movements.append(text)

        return movements

    def scrape_team_page(self, url):
        """Extrae información de una página de equipo en Liquipedia."""
        print(f"   [!] Abriendo navegador para: {url}")
        try:
            # Crear driver nuevo para cada URL
            self.driver = self._create_driver()
            self.driver.set_page_load_timeout(20)
            
            self.driver.get(url)
            # Hacer scroll hacia abajo para disparar la carga de elementos dinámicos
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # Activar "Show All" si existe y no está activo
            try:
                show_all_li = self.driver.find_element(By.XPATH, "//li[contains(@class, 'show-all')]")
                if 'active' not in show_all_li.get_attribute('class'):
                    # Hacer scroll hasta el elemento para que sea clickable
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_all_li)
                    time.sleep(1)
                    show_all_link = show_all_li.find_element(By.TAG_NAME, 'a')
                    # Usar JavaScript click para evitar intercepciones
                    self.driver.execute_script("arguments[0].click();", show_all_link)
                    time.sleep(2)  # Esperar a que cargue el contenido
            except Exception as e:
                print(f"   [!] No se pudo activar 'Show All': {e}")
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            print(f"   [!] Error al cargar página: {e}")
            return None
        finally:
            # Limpiar driver
            self.cleanup()
        
        if not soup:
            return None

        full_text = self._extract_first_paragraph(soup)
        infobox = self._extract_infobox(soup)
        name = self._extract_title(soup)
        movements = self.extract_movements(soup)
        tournaments = self._extract_tournaments(soup)

        return {
            "type": "team",
            "url": url,
            "name": name,
            "infobox": infobox,
            "full_text": full_text,
            "movements": movements,
            "tournaments": tournaments,
            "extracted_at": datetime.now().isoformat(),
            "combined_text": f"{name}. {full_text}",
        }

if __name__ == "__main__":
    print("Este módulo se usa desde chatbot/ingest.py.")
