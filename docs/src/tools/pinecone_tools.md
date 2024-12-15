# Pinecone Tools

The PineconeTools class provides a comprehensive set of methods to interact with the Pinecone vector database API. It offers powerful functionality for creating and managing indexes, upserting and querying vectors, and handling vector metadata. This class is designed to simplify the process of working with the Pinecone API, handling authentication and request formatting internally.

### Class Methods

##### __init__(api_key: str = None)

Initializes the PineconeTools class with the Pinecone API key. If not provided, it attempts to use the PINECONE_API_KEY environment variable.

##### create_index(name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1")

Creates a new Pinecone index with the specified parameters.

```python
PineconeTools().create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    cloud="aws",
    region="us-east-1"
)
```

##### delete_index(name: str)

Deletes a Pinecone index with the given name.

```python
PineconeTools().delete_index("my-index")
```

##### list_indexes()

Lists all available Pinecone indexes.

```python
indexes = PineconeTools().list_indexes()
print(indexes)
```

##### upsert_vectors(index_name: str, vectors: List[Dict[str, Any]])

Upserts vectors into a Pinecone index.

```python
vectors = [
    {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"key": "value"}},
    {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"key": "value2"}}
]
PineconeTools().upsert_vectors("my-index", vectors)
```

##### query_index(index_name: str, query_vector: List[float], top_k: int = 10, filter: Dict = None, include_metadata: bool = True)

Queries a Pinecone index for similar vectors.

```python
query_vector = [0.1, 0.2, 0.3]
results = PineconeTools().query_index("my-index", query_vector, top_k=5)
print(results)
```

##### delete_vectors(index_name: str, ids: List[str])

Deletes vectors from a Pinecone index by their IDs.

```python
PineconeTools().delete_vectors("my-index", ["vec1", "vec2"])
```

##### update_vector_metadata(index_name: str, id: str, metadata: Dict[str, Any])

Updates the metadata of a vector in a Pinecone index.

```python
new_metadata = {"key": "updated_value"}
PineconeTools().update_vector_metadata("my-index", "vec1", new_metadata)
```

##### describe_index_stats(index_name: str)

Gets statistics about a Pinecone index.

```python
stats = PineconeTools().describe_index_stats("my-index")
print(stats)
```

##### normalize_vector(vector: List[float])

A static method that normalizes a vector to unit length.

```python
normalized_vector = PineconeTools.normalize_vector([1.0, 2.0, 3.0])
print(normalized_vector)
```

##### get_pinecone_index(name: str)

Returns a Pinecone index object for the given index name.

```python
index = PineconeTools().get_pinecone_index("my-index")
```

### Error Handling

All methods in the PineconeTools class include error handling. If an operation fails, an exception will be raised with a descriptive error message. It's recommended to wrap calls to these methods in try-except blocks to handle potential errors gracefully.

### Environment Variables

The `PineconeTools` class prioritizes the use of the `PINECONE_API_KEY` environment variable. If you prefer not to pass the API key explicitly when initializing the class, ensure this environment variable is set:

```bash
export PINECONE_API_KEY="your-api-key-here"
```

### Additional Usage Examples

Here are some more advanced usage examples:

```python
# Create an index with custom parameters
PineconeTools().create_index(
    name="custom-index",
    dimension=768,
    metric="dotproduct",
    cloud="gcp",
    region="us-central1"
)

# Query with metadata filter
results = PineconeTools().query_index(
    "my-index",
    query_vector=[0.1, 0.2, 0.3],
    top_k=5,
    filter={"category": "electronics"}
)

# Batch delete vectors
PineconeTools().delete_vectors("my-index", ["id1", "id2", "id3"])

# Get index statistics
stats = PineconeTools().describe_index_stats("my-index")
print(f"Total vector count: {stats['total_vector_count']}")
```

These examples demonstrate more complex operations and show how to use some of the additional parameters available in the methods.

### Usage Notes

To use the PineconeTools class, you must set the PINECONE_API_KEY environment variable or provide the API key when initializing the class. These credentials are essential for authenticating with the Pinecone API and are securely managed by the class.

The class methods handle API authentication internally, abstracting away the complexity of token management. This allows developers to focus on making API calls and processing the returned data without worrying about the underlying authentication mechanism.

All methods in the PineconeTools class return data in the form of Python dictionaries or lists, making it easy to work with the results in your application. The structure of the returned data closely mirrors the JSON responses from the Pinecone API, ensuring that you have access to all the details provided by the API.

Error handling is built into these methods, with exceptions being caught and re-raised with additional context. This helps in debugging and handling potential issues that may arise during API interactions.

Here's an example of how you might create a vector database agent using these tools:

```python
vector_db_agent = Agent(
    role="Vector Database Manager",
    goal="Manage and query vector databases efficiently",
    attributes="Knowledgeable about vector databases, detail-oriented, efficient in data management",
    tools={
        PineconeTools.create_index,
        PineconeTools.upsert_vectors,
        PineconeTools.query_index,
        PineconeTools.delete_vectors,
        PineconeTools.update_vector_metadata,
        PineconeTools.describe_index_stats
    },
    llm=OpenrouterModels.haiku
)

def manage_vector_db(agent, action, **kwargs):
    return Task.create(
        agent=agent,
        instruction=f"Perform the following action on the vector database: {action}",
        context=f"Action parameters: {kwargs}"
    )

# Example usage
response = manage_vector_db(vector_db_agent, "create_index", name="my-index", dimension=1536)
print(response)

response = manage_vector_db(vector_db_agent, "upsert_vectors", index_name="my-index", vectors=[...])
print(response)

response = manage_vector_db(vector_db_agent, "query_index", index_name="my-index", query_vector=[...])
print(response)
```

This vector database agent can leverage the Pinecone tools to manage and query vector databases, making it a powerful assistant for tasks involving vector embeddings and similarity search.