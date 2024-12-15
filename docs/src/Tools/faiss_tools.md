# Local Vector Storage

The FAISSTools class provides a comprehensive set of methods to interact with Facebook AI Similarity Search (FAISS) for efficient similarity search and clustering of dense vectors. It offers powerful functionality for creating, managing, and querying FAISS indexes, which are particularly useful for tasks involving large-scale vector similarity searches, such as semantic search, recommendation systems, and more. 

### Class Methods

##### __init__(dimension: int, metric: str = "IP")

Initializes the FAISSTools with the specified vector dimension and distance metric. The default metric is "IP" (Inner Product).

```python
faiss_tools = FAISSTools(dimension=768, metric="IP")
```

Note: This method includes error handling for importing the `faiss` library. If `faiss` is not installed, it will raise an `ImportError` with instructions on how to install it.

##### create_index(index_type: str = "Flat")

Creates a new FAISS index of the specified type. Currently supports "Flat" index type with both "IP" (Inner Product) and "L2" (Euclidean) metrics.

```python
faiss_tools.create_index(index_type="Flat")
```

##### load_index(index_path: str)

Loads a FAISS index and its associated metadata from files. The method automatically appends the `.faiss` extension for the index file and `.metadata` for the metadata file.

```python
faiss_tools.load_index("/path/to/product_embeddings")
```

##### save_index(index_path: str)

Saves the current FAISS index and its metadata to files. The method automatically appends the `.faiss` extension for the index file and `.metadata` for the metadata file.

```python
faiss_tools.save_index("/path/to/save/product_embeddings")
```

##### add_vectors(vectors: np.ndarray)

Adds vectors to the FAISS index. Automatically normalizes vectors if using Inner Product similarity.

```python
vectors = np.random.rand(100, 768)  # 100 vectors of dimension 768
faiss_tools.add_vectors(vectors)
```

##### search_vectors(query_vectors: np.ndarray, top_k: int = 10)

Searches for similar vectors in the FAISS index, returning the top-k results.

```python
query = np.random.rand(1, 768)  # 1 query vector of dimension 768
distances, indices = faiss_tools.search_vectors(query, top_k=5)
```

##### remove_vectors(ids: np.ndarray)

Removes vectors from the FAISS index by their IDs.

```python
ids_to_remove = np.array([1, 3, 5])
faiss_tools.remove_vectors(ids_to_remove)
```

##### get_vector_count()

Returns the number of vectors in the FAISS index.

```python
count = faiss_tools.get_vector_count()
```

##### set_metadata(key: str, value: Any)

Sets metadata for the index.

```python
faiss_tools.set_metadata("description", "Product embeddings index")
```

##### get_metadata(key: str)

Retrieves metadata from the index.

```python
description = faiss_tools.get_metadata("description")
```

##### set_embedding_info(provider: str, model: str)

Sets the embedding provider and model information.

```python
faiss_tools.set_embedding_info("openai", "text-embedding-ada-002")
```

##### normalize_vector(vector: np.ndarray)

A static method that normalizes a vector to unit length. This is used internally for "IP" metric calculations.

```python
normalized_vector = FAISSTools.normalize_vector(vector)
```

### Understanding Index File Naming

When working with FAISS indexes using the FAISSTools class, the index files are saved with specific extensions. The main index file is saved with a `.faiss` extension, while the associated metadata is saved with a `.metadata` extension.

When using `save_index()` or `load_index()`, you should provide the base filename without any extension. The method will automatically append the correct extensions when saving or loading files.

For example, if you save an index as "product_embeddings":
- The main index file will be saved as "product_embeddings.faiss"
- The metadata file will be saved as "product_embeddings.metadata"

To load this index later, you would use:

```python
faiss_tools.load_index("/path/to/product_embeddings")
```

The method will automatically look for "product_embeddings.faiss" and "product_embeddings.metadata" files.

### Usage Notes

To use the FAISSTools class, you need to have FAISS installed. You can install it using pip:

```bash
pip install faiss-cpu  # for CPU-only version
# or
pip install faiss-gpu  # for GPU support
```

The FAISSTools class is designed to work with numpy arrays for vector operations. Make sure you have numpy installed and imported in your project.

When using Inner Product (IP) similarity, vectors are automatically normalized to unit length before being added to the index or used for querying. This ensures consistent similarity calculations.

Error handling is built into these methods, with appropriate exceptions being raised for common issues such as dimension mismatches or file not found errors.

Here's an example of how you might create a semantic search agent using these tools:

[triplebacktick]python
semantic_search_agent = Agent(
    role="Semantic Search Expert",
    goal="Perform efficient and accurate semantic searches on large datasets",
    attributes="Knowledgeable about vector embeddings and similarity search algorithms",
    tools={FAISSTools.quer},
    llm=OpenrouterModels.haiku
)

def semantic_search_task(agent, query, index_path):
    return Task.create(
        agent=agent,
        context=f"FAISS index path: {index_path}\nQuery: {query}",
        instruction="Load the FAISS index, perform a semantic search for the given query, and return the top 5 most similar results."
    )

# Usage
index_path = "/path/to/your/specific_index_name"
query = "Example search query"
results = semantic_search_task(semantic_search_agent, query, index_path)
print(results)
[triplebacktick]

This semantic search agent can leverage the FAISSTools to load pre-built indexes, perform similarity searches, and return relevant results based on vector embeddings.

### Best Practices

1. **Index Creation**: Choose the appropriate index type based on your dataset size and performance requirements. The "Flat" index is suitable for small to medium-sized datasets, but for larger datasets, consider using more advanced index types provided by FAISS.

2. **Vector Normalization**: When using Inner Product similarity, vectors are automatically normalized. However, if you're using L2 distance, consider normalizing your vectors before adding them to the index for consistent results.

3. **Metadata Management**: Use the metadata functionality to store important information about your index, such as the embedding model used, dataset description, or any other relevant details.

4. **Error Handling**: Always handle potential exceptions, especially when loading indexes or performing searches with user-provided queries.

5. **Performance Optimization**: For large-scale applications, consider using GPU-enabled FAISS and experiment with different index types and parameters to optimize performance.

6. **Index Persistence**: Regularly save your index to disk, especially after adding or removing vectors, to ensure data persistence and quick recovery in case of system failures.

By following these practices and leveraging the full capabilities of the FAISSTools class, you can build powerful and efficient similarity search systems within your Orchestra projects.