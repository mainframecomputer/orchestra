# Embeddings Tools

The EmbeddingsTools class provides a set of methods for generating embeddings using various AI platforms, including OpenAI, Cohere, and Mistral AI. It allows users to easily generate vector representations of text data for use in natural language processing tasks such as semantic search, clustering, and classification.

### Class Methods

##### get_openai_embeddings()

Generates embeddings for the given input text using OpenAI's API. It takes the input text (either a single string or a list of strings) and the desired model as arguments, and returns a tuple containing the embeddings and the number of dimensions for the chosen model.

```python
EmbeddingsTools.get_openai_embeddings(
    input_text="This is a sample text.",
    model="text-embedding-ada-002"
)
```

##### get_cohere_embeddings()

Generates embeddings for the given input text using Cohere's API. It takes the input text (either a single string or a list of strings), the desired model, and the input type as arguments, and returns a tuple containing the embeddings and the number of dimensions for the chosen model.

```python
EmbeddingsTools.get_cohere_embeddings(
    input_text=["Text 1", "Text 2"],
    model="embed-english-v3.0",
    input_type="search_document"
)
```

##### get_mistral_embeddings()

Generates embeddings for the given input text using Mistral AI's API. It takes the input text (either a single string or a list of strings) and the desired model as arguments, and returns a tuple containing the embeddings and the number of dimensions for the chosen model.

```python
EmbeddingsTools.get_mistral_embeddings(
    input_text=["Text 1", "Text 2"],
    model="mistral-embed"
)
```

### Usage Notes

To use the EmbeddingsTools class, you need to have valid API keys for the respective AI platforms. The API keys should be set as environment variables:

The class methods automatically handle the API requests and return the embeddings along with the number of dimensions for the chosen model. The input text can be provided as a single string or a list of strings, allowing you to generate embeddings for multiple texts in a single call.

The available models for each platform are defined in the MODEL_DIMENSIONS dictionary within the EmbeddingsTools class. You can choose the desired model by passing its name as the model argument to the respective method.

The generated embeddings are returned as a list of lists, where each inner list represents the embedding vector for a single input text. The number of dimensions for the chosen model is also returned as part of the tuple.

In case of any errors during the API requests, the methods will raise appropriate exceptions with detailed error messages. Make sure to handle these exceptions in your code and provide appropriate error handling and logging mechanisms.

It's important to note that generating embeddings can be computationally expensive, especially for large datasets. Consider the cost implications and rate limits of the respective AI platforms when using these methods in your applications.

The EmbeddingsTools class provides a convenient way to generate embeddings from different AI platforms using a consistent interface. You can easily switch between platforms by calling the appropriate method and providing the necessary arguments.

Remember to keep your API keys secure and avoid sharing them publicly. It's recommended to store them as environment variables or in a secure configuration file.

