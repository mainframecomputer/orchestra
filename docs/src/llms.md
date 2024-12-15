# LLMs

Mainframe-Orchestra supports integrating a variety of language models, providing developers with the flexibility to choose the most appropriate model for their specific use case. The Language Model (LLM) interfaces in Orchestra offer a unified and consistent way to interact with various AI models from different providers.

### LLM Interface Structure

The LLM interfaces are defined in the llm.py module. This module contains several classes, each representing a different LLM provider:

- OpenaiModels
- AnthropicModels
- OpenrouterModels
- OllamaModels
- GroqModels

Each class contains static methods corresponding to specific models offered by the provider, following a consistent structure.

### Supported Language Models

Orchestra supports a wide range of language models from various providers. Here's an overview of some supported models:

- OpenAI Models: GPT-3.5 Turbo, GPT-4 Turbo, GPT-4, GPT-4o
- Anthropic Models: Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku, Claude-3.5 Sonnet
- Openrouter Models: Various models including Anthropic Claude, OpenAI GPT, Llama, Mistral AI, and more
- Ollama Models: Llama 3, Gemma, Mistral, Qwen, Phi-3, Llama 2, CodeLlama, LLaVA, Mixtral
- Groq Models: Gemma, Llama3, Llama3.1, Mixtral

### Integrating Language Models

To integrate a language model within a `Task` object, you simply need to specify the appropriate model function from the corresponding class. Here's an example:

```python
from mainframe_orchestra import OpenrouterModels

llm = OpenrouterModels.haiku
```

In this example, we're using the OpenrouterModels.haiku model. You can then assign llm to any agent. The llm parameter is passed to the Task.create() method, allowing the task to use the specified language model for generating responses.

### Language Model Selection Considerations

When selecting a language model for your agents, consider the following factors:

- Performance: Different models vary in speed, accuracy, and output quality.
- Cost: Consider the cost implications, especially for production deployments.
- Capabilities: Ensure the selected model aligns with your task requirements (e.g., natural language generation, code generation).
- Context Window Size: For tasks requiring larger context, choose models with appropriate context window sizes.
- Tool Use Capabilities: Some models are better suited for tasks involving tool use.

### Advanced Techniques

Orchestra supports several advanced techniques for working with language models:

- Chaining Multiple Models: You can use different models for different stages of a workflow.
- Model-Agnostic Tasks: Design tasks that can work with various language models by passing the LLM as a parameter.
- Custom Models: Use the `custom_model` method to work with models not explicitly defined.

### Using Custom Models

Orchestra provides flexibility to use custom or unsupported models through the `custom_model` method available for each provider. Here's how you can use it:

```python
# OpenAI custom model
llm = OpenaiModels.custom_model("gpt-5")

# Anthropic custom model
llm = AnthropicModels.custom_model("claude-3-opus-20240229")

# OpenRouter custom model
llm = OpenrouterModels.custom_model("meta-llama/llama-3-70b-instruct")

# Ollama custom model
llm = OllamaModels.custom_model("llama3")

```

This approach allows you to use models that may not have pre-built functions in Orchestra, or to easily switch between different versions or fine-tuned variants of models. Remember to ensure that you have the necessary API access and credentials for the custom model you're trying to use.

### Conclusion

The Orchestra framework provides a robust and flexible approach to integrating a wide range of language models. By allowing language model selection at the task level and providing a consistent interface across different providers, Orchestra empowers developers to optimize their AI workflows for specific use cases while maintaining code simplicity and reusability.

