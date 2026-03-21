import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

class LiquipediaScraper:
    def __init__(self, game):
        self.base_url = f"https://liquipedia.net/{game}/"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Chatbot Project/1.0)'}

    def fetch_tournament_results(self, tournament_path):
        url = self.base_url + tournament_path
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print("Error al acceder a Liquipedia")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ejemplo: Extraer tablas de posiciones (clase común en Liquipedia)
        tables = soup.find_all('table', {'class': 'wikitable'})
        data = []
        
        for table in tables:
            # Aquí conviertes la tabla HTML en un DataFrame de Pandas
            df = pd.read_html(str(table))[0]
            data.append(df.to_dict(orient='records'))
            
        return data

# Ejemplo de uso:
scraper = LiquipediaScraper('valorant')
resultados = scraper.fetch_tournament_results('VCT/2024/Champions')

# Guardar a JSON para el Chatbot
with open('data/resultados_vct.json', 'w') as f:
    json.dump(resultados, f, indent=4)