def embed_text(model, paragraphs):
    """
    Converts text to embeddings.
    
    """
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    print(embeddings)
    return embeddings