# LLMs

Mainframe-Orchestra supports integrating a variety of language models through a unified LiteLLM interface, providing developers with the flexibility to choose the most appropriate model for their specific use case while maintaining a consistent API across all providers.

### Unified LLM Architecture

Orchestra v1.0.0 introduces a major architectural change: all language model interactions are now handled through LiteLLM, a unified interface that supports 100+ LLM providers. This means:

- **Single dependency**: Instead of managing separate dependencies for each provider (anthropic, openai, groq, etc.), Orchestra now uses only LiteLLM
- **Consistent interface**: All models work the same way regardless of provider
- **Automatic fallback**: You can specify multiple models in a list for automatic failover
- **Simplified configuration**: One configuration approach for all providers

### LLM Interface Structure

The LLM interfaces are defined in the llm.py module using the unified `LiteLLMProvider` class. This module contains several model classes, each representing a different LLM provider:

- OpenaiModels
- AnthropicModels
- OpenrouterModels
- OllamaModels
- GroqModels
- TogetheraiModels
- GeminiModels
- DeepseekModels
- HuggingFaceModels

Each class contains static methods corresponding to specific models offered by the provider, all using the same underlying LiteLLM interface.

### Supported Language Models

Orchestra supports a wide range of language models from various providers through LiteLLM. Here's an overview of some supported models:

- OpenAI Models: GPT-4o, GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, o1, o3, o4
- Anthropic Models: Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku, Claude-3.5 Sonnet, Claude-3.7 Sonnet
- Openrouter Models: Various models including Anthropic Claude, OpenAI GPT, Llama, Mistral AI
- Ollama Models: Llama 3, Gemma, Mistral, Qwen, Phi-3, Llama 2, CodeLlama, LLaVA, Mixtral
- Groq Models: Gemma, Llama3, Llama3.1, Mixtral
- Togetherai Models: Meta Llama 3.1, Mixtral, Mistral, many other open source models
- Gemini Models: Gemini 2.0, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.5 Pro (Flash)
- Deepseek Models: Deepseek Reasoner, Deepseek Chat
- HuggingFace Models: Access to thousands of models via Hugging Face Inference API

### Integrating Language Models

To integrate a language model within a `Task` object, you simply need to specify the appropriate model function from the corresponding class. Here's an example:

```python
from mainframe_orchestra import OpenrouterModels

llm = OpenrouterModels.haiku
```

In this example, we're using the OpenrouterModels.haiku model. You can then assign llm to any agent. The llm parameter is passed to the Task.create_async() method, allowing the task to use the specified language model for generating responses. This is helpful if you want to use the same model for multiple agents. Alternatively, you can pass the model directly to the agent as a parameter, like `llm=OpenrouterModels.haiku`, if you want certain agents to use specific models.

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

### LLM Fallbacks via Lists

You can now specify multiple LLMs for a task, allowing for automatic fallback if the primary LLM fails or times out.

In this example, if `AnthropicModels.sonnet_3_5` fails (e.g., due to rate limiting), the task automatically falls back to `AnthropicModels.haiku_3_5`. You can specify as many LLMs as you want in the list and they will be tried in order. You can have the models fall back to another of the same provider, or you can have them fall back to a different provider if the provider itself fails. This is useful for handling rate limits or other failures that may occur with certain LLMs, particularly in a production environment.

```python
from mainframe_orchestra import Agent, GitHubTools, AnthropicModels

researcher = Agent(
    role="GitHub researcher",
    goal="find relevant Python agent repositories with open issues",
    attributes="analytical, detail-oriented, able to assess repository relevance and popularity",
    llm=[AnthropicModels.sonnet_3_5, AnthropicModels.haiku_3_5],
    tools={GitHubTools.search_repositories, GitHubTools.get_repo_details}
)
```

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

### Customizing OpenAI Base URL

Orchestra provides flexibility to customize the OpenAI base URL, allowing you to connect to OpenAI-compatible APIs or proxies. This is particularly useful for:

- Using Azure OpenAI endpoints
- Connecting to local deployments of OpenAI-compatible models
- Using proxy services that implement the OpenAI API
- Working with custom OpenAI-compatible endpoints

There are three ways to customize the OpenAI base URL:

#### 1. Using Environment Variables

Set the `OPENAI_BASE_URL` environment variable before running your application:

```python
import os
os.environ["OPENAI_BASE_URL"] = "https://your-custom-endpoint.com/v1"

# Now all OpenAI requests will use the custom base URL
from mainframe_orchestra import Agent, Task, OpenaiModels
```

#### 2. Setting a Global Base URL

Use the `set_base_url` class method to set a default base URL for all OpenAI requests:

```python
from mainframe_orchestra.llm import OpenaiModels

# Set a global base URL for all OpenAI requests
OpenaiModels.set_base_url("https://your-custom-endpoint.com/v1")

# All subsequent requests will use this base URL
response, error = await OpenaiModels.gpt_4o(messages=[{"role": "user", "content": "Hello"}])
```

#### 3. Per-Request Base URL

Specify a custom base URL for a specific request:

```python
from mainframe_orchestra.llm import OpenaiModels

# Use a custom base URL for this specific request only
response, error = await OpenaiModels.gpt_4o(
    messages=[{"role": "user", "content": "Hello"}],
    base_url="https://your-custom-endpoint.com/v1"
)
```

This flexibility allows you to easily switch between different OpenAI-compatible endpoints based on your specific needs, without changing your code structure.

### Benefits of the LiteLLM Architecture

The unified LiteLLM architecture in Orchestra v1.0.0 provides several key benefits:

- **Simplified Dependencies**: No need to install and manage separate packages for each LLM provider
- **Consistent Interface**: Same API patterns across all providers reduce learning curve
- **Automatic Parameter Handling**: LiteLLM automatically handles provider-specific parameters
- **Enhanced Reliability**: Built-in retry logic and error handling across all providers
- **Future-Proof**: Easy to add support for new providers as they become available

### Conclusion

The Orchestra framework provides a robust and flexible approach to integrating a wide range of language models through its unified LiteLLM interface. By consolidating all provider interactions into a single, consistent API while maintaining the ability to specify models at the task level, Orchestra empowers developers to optimize their AI workflows for specific use cases while maintaining code simplicity and reusability. The automatic fallback capabilities and simplified dependency management make it easier than ever to build reliable, production-ready AI applications.
