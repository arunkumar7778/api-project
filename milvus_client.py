from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def store_embeddings_in_milvus(embeddings, ids=None, collection_name="default_collection", host='localhost', port='19530'):
    """
    Connects to Milvus, deletes existing data in the specified collection (if any), 
    and inserts new embeddings.

    Args:
        embeddings (list or numpy.array): Embedding vectors to store.
        ids (list, optional): List of IDs associated with the embeddings (optional).
        collection_name (str): Name of the Milvus collection.
        host (str): Host address of the Milvus instance. Default is 'localhost'.
        port (str): Port of the Milvus instance. Default is '19530'.
    """
    # Connect to Milvus
    connections.connect("default", host=host, port=port)
    
    # Check if the collection exists
    if Collection.exists(collection_name):
        # Load the collection and delete all data
        collection = Collection(collection_name)
        collection.drop()
        print(f"Existing collection '{collection_name}' found and dropped.")
    else:
        print(f"No existing collection named '{collection_name}'. Creating a new one.")

    # Define the schema (assuming embeddings are vectors of a certain dimension)
    dim = len(embeddings[0])  # Assuming all embeddings have the same dimension
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Collection for storing embeddings")
    
    # Create the collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Prepare data for insertion (embeddings with optional IDs)
    data_to_insert = [ids, embeddings] if ids else [embeddings]
    
    # Insert the data into the collection
    collection.insert(data_to_insert)
    
    # Flush to ensure data is written
    collection.flush()
    
    print(f"New data successfully inserted into {collection_name}!")
