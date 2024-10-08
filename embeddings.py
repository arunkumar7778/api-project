# embed.py
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def split_content(content, chunk_size=512):
    """Split the content into chunks of specified size."""
    words = content.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def embed_content(chunks):
    """Embed content using Google Generative AI Embeddings."""
    model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key='AIzaSyDFvEzEBCcN8x56hfl5tB8mVK-Wvuij9qk'
    )
    
    # Generate embeddings (change this line to the correct method if necessary)
    embeddings = model.embed_documents(chunks)  # or the correct method name
    
    # Ensure the embeddings are in 2D shape
    if len(embeddings) > 0 and isinstance(embeddings[0], list):  # Check if it's a list of lists
        embeddings = np.array(embeddings)  # Convert to a NumPy array
    else:
        embeddings = np.array([embeddings])  # Wrap in an extra list to ensure 2D

    return embeddings.astype('float32')  # Ensure they are float32 for FAISS


if __name__ == "__main__":
    sample_content = "Your long content here."  # Replace with actual content
    chunks = split_content(sample_content)
    embeddings = embed_content(chunks)
    print(embeddings)  # Print the embeddings
