import requests
from bs4 import BeautifulSoup
import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from deep_translator import GoogleTranslator  # pip install deep-translator

class LiquipediaScraper:
    def __init__(self, game):
        self.base_url = f"https://liquipedia.net/{game}/"
        self.headers = {'User-Agent': 'Mozilla/5.0'}

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")

        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.translator = GoogleTranslator(source='auto', target='es')  # Traductor al español

    def get_soup(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            return BeautifulSoup(response.text, 'html.parser')
        except:
            return None

    def extract_external_references(self, soup):
        refs = []

        for a in soup.find_all('a', class_='external text'):
            refs.append({
                "label": a.get_text(strip=True),
                "url": a.get('href')
            })

        return refs

    def scrape_twitter(self, url):
        try:
            self.driver.get(url)

            # Esperar texto del tweet
            tweet = self.wait.until(
                EC.presence_of_element_located((By.XPATH, '//div[@data-testid="tweetText"]'))
            )

            #  Buscar fecha
            time_element = self.wait.until(
                EC.presence_of_element_located((By.XPATH, '//time'))
            )

            tweet_text = tweet.text
            tweet_date = time_element.get_attribute("datetime")

            # Traducir al español
            tweet_text_es = self.traducir(tweet_text)

            return {
                "text": tweet_text,
                "text_es": tweet_text_es,
                "date": tweet_date
            }

        except Exception as e:
            print(f"Error Twitter: {url} -> {e}")
            return None

    def scrape_generic(self, url):
        try:
            self.driver.get(url)
            time.sleep(3)

            paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
            texts = [p.text for p in paragraphs if len(p.text) > 30]

            content = " ".join(texts[:10])
            content_es = self.traducir(content)

            return {
                "text": content,
                "text_es": content_es
            }

        except Exception as e:
            print(f"Error scrapeando genérico: {url} -> {e}")
            return None

    def scrape_external_content(self, url):
        if "twitter.com" in url or "x.com" in url:
            return self.scrape_twitter(url)
        return self.scrape_generic(url)

    def traducir(self, texto):
        if not texto:
            return None
        try:
            return self.translator.translate(texto)
        except Exception as e:
            print(f"Error traduciendo: {e}")
            return texto

    def scrape_page(self, page):
        soup = self.get_soup(self.base_url + page)
        if not soup:
            return []

        refs = self.extract_external_references(soup)
        results = []

        for ref in refs:
            print(f"Scrapeando: {ref['url']}")
            content = self.scrape_external_content(ref['url'])

            # Guardar con traducción
            if content is None:
                results.append({
                    "label": ref['label'],
                    "url": ref['url'],
                    "content": None,
                    "content_es": None,
                    "date": None
                })
            else:
                results.append({
                    "label": ref['label'],
                    "url": ref['url'],
                    "content": content.get("text"),
                    "content_es": content.get("text_es"),
                    "date": content.get("date")  # solo aplica a Twitter
                })

        return results

# --- USO MULTIJUEGOS Y EQUIPOS ---
games_teams = {
    "fortnite": ["G2_Esports", "NinjasInPyjamas", "FaZeClan"],
    "valorant": ["Sentinels", "Fnatic", "TeamLiquid"],
    "csgo": ["Astralis", "FURIA", "G2_Esports"]
}

all_data = []
scraper = LiquipediaScraper(None)

for game, teams in games_teams.items():
    print(f"\n--- Scrapeando juego: {game} ---")
    scraper.base_url = f"https://liquipedia.net/{game}/"

    for team in teams:
        print(f"\nScrapeando equipo: {team}")
        try:
            data = scraper.scrape_page(team)
            for item in data:
                item["game"] = game
                item["team"] = team
            all_data.extend(data)
        except Exception as e:
            print(f"Error scrapeando {team} en {game}: {e}")

# Guardar todo en un JSON
with open("external_refs_multiple_es.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)
