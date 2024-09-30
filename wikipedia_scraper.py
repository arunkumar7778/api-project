import requests
from bs4 import BeautifulSoup
import re

def scrape_wikipedia(url: str):
    """
    Extracts the main content from a Wikipedia page and removes numbers.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the page. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find(id="mw-content-text").find_all('p')
    
    # Extract text from paragraphs, clean it, and remove numbers
    paragraphs = [
        re.sub(r'\[\d+\]', '', p.get_text().strip())  # Remove numbers in square brackets
        for p in content if p.get_text().strip()
    ]
    return paragraphs