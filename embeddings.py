from wikipedia_scraper import scrape_wikipedia
from sentence_transformers import SentenceTransformer  # type: ignore

def embed_text(model, paragraphs):
    """
    Converts text to embeddings.
    
    Parameters:
    - model: The model used for generating embeddings.
    - paragraphs: List of text paragraphs to be embedded.

    Returns:
    - embeddings: Tensor containing the embeddings of the input paragraphs.
    """
    if not paragraphs:
        raise ValueError("The paragraphs list is empty. Please provide valid text.")
    
    try:
        embeddings = model.encode(paragraphs, convert_to_tensor=True)
        print(embeddings)  # For debugging; remove in production
        return embeddings
    except Exception as e:
        raise RuntimeError(f"An error occurred while embedding the text: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Load your embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Use your desired model

        # Scrape Wikipedia
        url = "https://en.wikipedia.org/wiki/Nobita_Nobi"
        paragraphs = scrape_wikipedia(url)

        # Ensure the paragraphs are formatted correctly
        paragraphs = [p for p in paragraphs if isinstance(p, str) and p.strip()]

        # Embed the scraped text
        embeddings = embed_text(model, paragraphs)

        # You can now use `embeddings` for further processing
    except Exception as e:
        print(f"An error occurred: {e}")
