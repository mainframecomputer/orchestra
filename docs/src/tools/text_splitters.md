# Text Splitters

The SentenceSplitter class is a simple wrapper around the [sentence_splitter](https://github.com/mediacloud/sentence-splitter) library, providing an easy way to split text into overlapping chunks of sentences.

## Usage

```python
from mainframe_orchestra.tools import SentenceSplitter

splitter = SentenceSplitter(language='en')
chunks = splitter.split_text_by_sentences(
    text="Your long text here...",
    chunk_size=5,
    overlap=1
)
```

## Parameters

- `language`: The language of the text (default: 'en')
- `text`: The input text to split
- `chunk_size`: Number of sentences per chunk (default: 5)
- `overlap`: Number of sentences to overlap between chunks (default: 1)

## Return Value

Returns a list of strings, where each string is a chunk of text containing the specified number of sentences.

## Note

For more advanced usage or language-specific options, consider using the sentence_splitter library directly.


# Semantic Splitter

The SemanticSplitter class provides a sophisticated method for splitting text into semantically coherent chunks. This tool is particularly useful for processing large texts, preparing data for summarization, or creating more manageable segments for further NLP tasks. It uses sentence embeddings and community detection algorithms to group similar sentences together.

### Class Methods

##### chunk_text()

This is the main static method of the SemanticSplitter class. It takes a text input and returns a list of semantically coherent chunks.

```python
@staticmethod
def chunk_text(text: str, rearrange: bool = False, 
               embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small") -> List[str]:
```

Parameters:
- `text`: The input text to be split into chunks.
- `rearrange`: If True, sentences will be grouped by their semantic similarity, potentially changing their original order. When False, it maintains the original order of the text while still grouping similar sentences together.
- `embedding_provider`: The provider of the embedding model (e.g., "openai", "cohere", "mistral").
- `embedding_model`: The specific embedding model to use.

### Usage Notes

To use the SemanticSplitter class, you need to have the necessary API keys set up in your environment variables for the chosen embedding provider.

The class uses the sentence_splitter library for initial text segmentation and the igraph and leidenalg libraries for community detection. Make sure these dependencies are installed in your environment.

The SemanticSplitter process involves creating sentence segments, embedding them, detecting communities using graph algorithms, and finally creating chunks from these communities.

The class includes a method to split oversized communities, which can help manage very large chunks of text and ensure more balanced output.

Basic error handling is implemented, such as returning a single community for very short inputs.

The SemanticSplitter can be particularly useful when working with large text strings that need to be processed in smaller, semantically coherent chunks. This can improve the performance of downstream NLP tasks such as summarization, question-answering, or topic modeling.

### Advanced Usage

The SemanticSplitter offers several parameters that can be tuned for optimal performance:

- The `rearrange` parameter allows for reordering of sentences based on their semantic similarity. This can be useful for certain applications but should be used cautiously if preserving the original text order is important. Essentially, it will regroup sentences that are similar into a single chunk. If a sentence or line at the end of the document is similar to the first few, they will be regrouped together.

- Different embedding providers and models can be used by specifying the `embedding_provider` and `embedding_model` parameters. This allows for flexibility in choosing the most appropriate embedding method for your specific use case.

```python
# Example of advanced usage with custom parameters
chunks = SemanticSplitter.chunk_text(
    text=large_text,
    rearrange=True,
    embedding_provider="cohere",
    embedding_model="embed-english-v3.0"
)
```

This advanced usage demonstrates how to customize the chunking process for specific needs, such as using a different embedding provider or adjusting the chunking parameters.

### Performance Considerations

The SemanticSplitter can be computationally intensive, especially for very large texts. The performance is primarily affected by:

1. The length of the input text (number of sentences).
2. The chosen embedding model and provider.

For extremely large texts, consider breaking the text into individual documents or subsections before applying the SemanticSplitter.

### Use with Knowledgebases

The SemanticSplitter can be particularly useful when working with knowledge bases or large documents that need to be processed in smaller, semantically coherent chunks. This can improve the performance of downstream NLP tasks such as summarization, question-answering, or topic modeling. Articles, documents can be fed through the splitter to feed the chunks into an agent knowledgebase.

By leveraging the SemanticSplitter tool, agents can efficiently process and analyze large volumes of text, breaking them down into manageable, semantically coherent chunks for further processing or analysis.

### Example Usage with Different Embedding Providers

Here's how to use the SemanticSplitter with different embedding providers:

```python
text = "This is a test text to demonstrate the semantic splitter. It should be split into meaningful chunks based on the content and similarity threshold. There are many different embedding providers and models available."

# Using OpenAI (default)
chunks = SemanticSplitter.chunk_text(text)

# Using Cohere
chunks = SemanticSplitter.chunk_text(text, embedding_provider="cohere", embedding_model="embed-english-v3.0")

# Using Mistral
chunks = SemanticSplitter.chunk_text(text, embedding_provider="mistral", embedding_model="mistral-embed")
```

These examples demonstrate how to use different embedding providers and models with the SemanticSplitter.