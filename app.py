from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from wikipedia_scraper import scrape_wikipedia
from embeddings import embed_text
from milvus_client import create_collection, insert_embeddings
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()
print("FAST API:", app)

# Create a Milvus collection and load the Sentence Transformer model
collection_name = "wiki_embeddings"
collection = create_collection(collection_name)

# Load SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

class WikipediaLoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

@app.post("/load")
async def load_wikipedia_data(request: WikipediaLoadRequest):
    try:
        # Scrape the Wikipedia page
        paragraphs = scrape_wikipedia(request.url)
        
        # Embed the scraped content
        embeddings = embed_text(sentence_model, paragraphs)
        
        # Insert into Milvus
        insert_embeddings(collection, embeddings, paragraphs)
        return {"status": "success", "message": f"Data from {request.url} loaded successfully."}
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(request: QueryRequest):
    try:
        # Step 1: Embed the question
        question_embedding = sentence_model.encode(request.question)
        print("Question embedding: ", question_embedding)
        # Step 2: Search Milvus for relevant data
        results = collection.search(    
            [question_embedding],
            "embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3,
            output_fields = ['text']
        )
        
        print("Search results: ", results)

        # Step 3: Collect the most relevant paragraphs
        relevant_texts = [result.entity.get("text") for result in results[0]]
        print("Relevant texts:", relevant_texts)  

        if not relevant_texts:
            return {"answer": "No relevant information found."}

        # Step 4: Use T5 for question answering
        
        input_text = f"question: {request.question} context: {','.join(relevant_texts)}"
        print("Input text to model:", input_text)  
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate an answer with T5
        output = t5_model.generate(input_ids, max_new_tokens=50)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        if answer:
            print(answer)
        else:
            print("No answer generated for context:", relevant_texts)  

        # Combine answers and return
        # combined_answer = " ".join(answers).strip() or "No answer generated."
        return {"answer": answer}
    except Exception as e:
        print("Error occurred:", str(e))  
        raise HTTPException(status_code=500, detail=str(e))