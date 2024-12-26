# FAQ

This FAQ page addresses common questions about setting up and using Mainframe-Orchestra. If you have additional questions, please feel free to open an issue on [our GitHub repository](https://github.com/mainframecomputer/orchestra).

### Setting Up Agents and Tasks

##### How do I set up an agent?

There are two main ways to set up agents in Orchestra:

One way to set up agents is by creating a class of agents. This approach allows you to group related agents together and provides a structured way to organize your agents.

```python
# Option 1: Class of agents
class Agents:
    web_researcher = Agent(
        role="web researcher",
        goal="find relevant information on the web",
        attributes="detail-oriented, analytical",
        llm=OpenrouterModels.haiku
    )
    summarizer = Agent(
        role="summarizer",
        goal="condense information into concise summaries",
        attributes="concise, clear communicator",
        llm=OpenrouterModels.haiku
    )
    programmer = Agent(
        role="programmer",
        goal="write and debug code",
        attributes="logical, problem-solver",
        llm=OpenrouterModels.haiku
    )
```

Another way to set up agents is by creating them directly in your script. This approach is more straightforward and can be useful for simpler setups or when you need to create agents on the fly.

```python
# Option 2: Direct setup in script
researcher = Agent(
    role="researcher",
    goal="conduct thorough research on given topics",
    attributes="analytical, detail-oriented",
    llm=OpenrouterModels.haiku
)
```

##### How do I set up a task?

To set up a task, you can wrap the Task.create method in a function that imports what it needs to perform the task. This makes the task modular and easy to use in sequences:

```python
from mainframe_orchestra import Task, OpenrouterModels

def research_task(topic):
    result = Task.create(
        agent="web_researcher",
        context=f"The user wants information about {topic}",
        instruction=f"Explain {topic} and provide a comprehensive summary",
        llm=OpenrouterModels.haiku
    )
    return result

# Usage
topic = "artificial intelligence"
research_result = research_task(topic)
```

### Language Models and Customization

##### How do I use a model that's not supported in the LLM function list?

You can use the custom_model method available for each provider to use models that are not explicitly listed in the LLM function list:

```python
from mainframe_orchestra import OpenrouterModels, AnthropicModels, OllamaModels, OpenaiModels

# OpenRouter custom model
llm = OpenrouterModels.custom_model(model_name="meta-llama/llama-3-70b-instruct")

# Anthropic custom model
llm = AnthropicModels.custom_model(model_name="claude-3-opus-20240229")

# Ollama custom model
llm = OllamaModels.custom_model(model_name="llama3.1:405b")

# OpenAI custom model
llm = OpenAIModels.custom_model(model_name="gpt-4o-mini")
```

### Privacy and Local Usage

##### How do I use Orchestra locally and privately?

To use Orchestra locally and privately:

- Download and install Ollama
- Pull the model you want to use
- Host it from the terminal
- Select the model function from llm.py or set a newer model card with the custom_model function

All requests will stay internal to your device in this configuration, ensuring privacy and local usage.

### Future Development

##### Are you adding more tools? Are you open to requests?

Yes, we are open to adding new features and toolkits as needed. If you have a request for a new tool or feature, please open an issue on [our GitHub repository](https://github.com/mainframecomputer/orchestra) or contribute directly to the project.

### Additional Questions

##### How do I handle errors when using tools?

Implement proper error handling when using tools, especially those that interact with external APIs. Tools that print and return errors rather than fail are preferable, as they can be provided back to the agent in a retry loop, allowing multiple attempts to succeed.

##### How should I manage API keys for various tools?

Use environment variables to manage API keys required by various tools. This enhances security and makes it easier to deploy your application across different environments.

##### Can I create custom tools?

Yes, you can create custom tools following the same pattern as the built-in ones. This allows you to extend Orchestra's functionality to meet your specific needs.

