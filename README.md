FastAPI: For building the web service.
Pydantic: For data validation.
wikipedia-scraper: For scraping Wikipedia content.
Sentence Transformers: For creating text embeddings.
Milvus: For storing and searching embeddings.
Transformers: For the T5 model used in question answering.
PyTorch: For running the models.

Wikipedia Scraper Module:
This module provides functionality to scrape the main content from Wikipedia pages, focusing on extracting text from paragraphs while removing unnecessary elements, such as reference numbers.
Features:
- Extracts main content from a specified Wikipedia page.
- Cleans the extracted text by removing numbers and unnecessary formatting.

Embedding Module:
This module provides functionality for generating text embeddings using state-of-the-art models, enabling efficient semantic search and natural language processing tasks.
Features:
- Generates embeddings for text using the Sentence Transformer model.
- Easy integration with various data sources, such as Wikipedia content.

Milvus Embedding Module:
This module provides functionality to interact with Milvus, a high-performance vector database, for storing and querying embeddings. It includes functions for creating a collection and inserting embeddings along with their corresponding text.
Features:
- Create a collection in Milvus for storing embeddings and associated text.
- Insert embeddings into the collection.
- Automatically manage existing collections by dropping them if they already exist.
- Create and load an index on the embedding field for efficient searching.

FastAPI Wikipedia Query Service:
This FastAPI application allows users to scrape Wikipedia pages, embed their content, and perform question answering based on the embedded data. It combines various NLP technologies to provide an efficient and effective query service.
Features
- Load Wikipedia Data: Scrapes content from a specified Wikipedia page and stores it as embeddings in a Milvus vector database.
- Query Data: Users can ask questions based on the loaded content, and the system provides answers using a T5 model.

Dependencies
The project requires the following libraries:
-fastapi: For building the API.
-beautifulsoup4: For scraping Wikipedia pages.
-requests: To make HTTP requests to Wikipedia.
-sentence-transformers: To generate embeddings for the content.
-pymilvus: To interact with Milvus for storing and querying embeddings.
-numpy: For handling vector calculations.
-langchain-google-genai: To interface with Google Generative AI (Gemini Pro).
