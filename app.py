from fastapi import FastAPI
from scrap import scrape_wikipedia_page
from embed import split_content, embed_content
from vector import store_embeddings_in_milvus, search_in_milvus
import numpy as np
from langchain_google_genai import GoogleGenerativeAI 

app = FastAPI()

@app.post("/scrape/")
async def scrape_and_store(url: str):
    """Scrape, embed, and store content from a Wikipedia page."""
    # Scrape content
    content = scrape_wikipedia_page(url)
    
    # Split and embed content
    chunks = split_content(content)
    embeddings = embed_content(chunks)
    
    # Store embeddings in Milvus
    store_embeddings_in_milvus(embeddings, chunks)
    
    return {"message": "Content scraped, embedded, and stored successfully."}

@app.post("/search/")
async def search_embeddings(query: str):
    """Search for the most similar embeddings given a query."""
    # Embed the query
    query_embedding = embed_content([query])[0]  # Get embedding for the query
    distances, results = search_in_milvus(query_embedding)
    
    # Define a threshold to determine if the result is relevant
    threshold = 0.7  # Adjust this based on your embedding model

    if not distances or np.min(distances) > threshold:
        # If no close match is found
        return {
            "message": "The answer to the question is not available in the provided content."
        }
    
    # Prepare a response based on the closest match
    closest_result = results[np.argmin(distances)]
    
    # Create a prompt for GoogleGenAI to answer only from the content
    prompt = (
        f"Based on the content: '{closest_result}', "
        f"answer the following question: '{query}'. "
        "If the answer is not found in the content, respond with 'The answer to this question is not available in the provided content.'"
    )
    
    # Use GoogleGenAI to generate an answer based on the prompt
    ai_response = GoogleGenerativeAI(model='geminipro',prompt=prompt)
    
    return {
        "distances": distances.tolist(),
        "result": closest_result,
        "ai_response": ai_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
