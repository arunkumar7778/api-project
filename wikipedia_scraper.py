import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_page(url):
    """Scrape content from a Wikipedia page."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find('div', {'class': 'mw-parser-output'}).get_text()
        return content
    else:
        raise Exception("Failed to retrieve page content.")

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Web_scraping"  # Example URL
    content = scrape_wikipedia_page(url)
    print(content[:500])  # Print the first 500 characters
