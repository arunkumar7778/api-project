from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

# Define Collection Schema for storing vectors
def create_collection(collection_name="wiki_embeddings"):
    # Connect to Milvus
    try:
        connections.connect(alias="default", host="127.0.0.1", port="19530", timeout=60)
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Drop the collection if it already exists
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    # Define the fields for the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
    ]

    # Define the schema
    schema = CollectionSchema(fields)

    # Create the collection with the new schema
    collection = Collection(name=collection_name, schema=schema)

    print(f"Collection '{collection_name}' created successfully!")
    return collection

def insert_embeddings(collection, embeddings, paragraphs):
    ids = list(range(1, len(embeddings) + 1))  # Generating simple incremental IDs
    entities = [ids, embeddings.tolist(), paragraphs]
    print(len(embeddings))
    print(len(paragraphs))
    print("Entities: ", entities)


    # Insert data into the collection
    try:
        collection.insert(entities)
        print("Inserted embeddings!")

        # Create an index on the embedding field
        index_params = {
            "index_type": "IVF_FLAT",   
            "metric_type": "L2",       
            "params": {"nlist": 128},   
        }

        # Create the index
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        print("Index created and collection loaded!")
    except Exception as e:
        print(f"Error during insert or index creation: {e}")