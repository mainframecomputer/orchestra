
# Introduction

Orchestra is a lightweight open-source agentic framework for creating LLM-driven task pipelines and multi-agent teams, centered around the concept of Tasks rather than conversation patterns.

[Orchestra GitHub Repository](https://github.com/mainframecomputer/orchestra)

### Core Principles

Orchestra is built around the concept of task completion, rather than conversation patterns. It has a modular architecture with interchangeable components. It's meant to be lightweight with minimal dependencies, and it offers transparency through a flat hierarchy and full prompt exposure.

![Orchestra Orchestrator](https://utfs.io/f/lKo6VaP8kaqVeFkvKtdZGxnUaslhq80BRH2VtP5O6oNbFvjw)

### Core Components

##### Tasks

Tasks are the fundamental building blocks of Orchestra. Each task represents a single, discrete unit of work to be performed by a Large Language Model. They include an optional context field for providing relevant background information, and an instruction that defines the core purpose of the task.

##### Agents

An Agent in Orchestra represents a specific role or persona with a clear goal. It can have optional attributes, and is powered by a selected language model (LLM). This structure allows Agents to maintain a consistent persona across multiple tasks. Agents can also be assigned tools, which are specific deterministic functions that the agent can use to interact with libraries, APIs, the internet, and more.

##### Tools

Tools in Orchestra are wrappers around external services or APIs, as well as utilities for common operations. You can link tools together with tasks to create structured, deterministic AI-integrated pipelines, offering precise control over the AI's actions in scenarios that require predictable workflows. Or, you can directly assign tools to agents, and the agents to tasks, enabling more autonomous, self-determined tool use. In this mode, AI Agents can independently choose and utilize tools to complete their assigned tasks.

##### Language Models

Orchestra supports various Language Models through a unified LiteLLM interface, providing access to models from OpenAI, Anthropic, Google, Groq, Together AI, and many others. This unified approach simplifies model switching and enables automatic fallback capabilities.

### Getting Started

To begin using Orchestra:

- Create a folder for your Orchestra projects
- Create a .env file with your relevant API Keys
- In your new folder, set up a virtual environment and install Orchestra

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install mainframe-orchestra
```

Once you have installed Orchestra, you can start building your agentic workflows and multi-agent teams.

### Simple Examples

#### Synchronous Usage (Simpler)
```python
# Single Agent example - Synchronous
from mainframe_orchestra import Task, Agent, WebTools, OpenrouterModels

researcher = Agent(
    role="research assistant",
    goal="answer user queries",
    attributes="thorough in web research",
    tools=[WebTools.serper_search],
    llm=OpenrouterModels.haiku
)

def research_task(agent, topic):
    return Task.create(
        agent=agent,
        instruction=f"Research {topic} and provide a summary of the top 3 results."
    )

def main():
    topic = input("Enter a topic to research: ")
    response = research_task(researcher, topic)
    print(response)

if __name__ == "__main__":
    main()
```

#### Asynchronous Usage (More Performant)
```python
# Single Agent example - Asynchronous
import asyncio
from mainframe_orchestra import Task, Agent, WebTools, OpenrouterModels

async def main():
    researcher = Agent(
        role="research assistant",
        goal="answer user queries",
        attributes="thorough in web research",
        tools=[WebTools.serper_search],
        llm=OpenrouterModels.haiku
    )

    async def research_task(agent, topic):
        return await Task.create_async(
            agent=agent,
            instruction=f"Research {topic} and provide a summary of the top 3 results."
        )

    topic = input("Enter a topic to research: ")
    response = await research_task(researcher, topic)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent Example

```python
# Multi Agent example - Showing both sync and async patterns
from mainframe_orchestra import Agent, Task, WebTools, WikipediaTools, AmadeusTools, OpenrouterModels, set_verbosity

set_verbosity(1)

# Define agents (same for both sync and async)
web_research_agent = Agent(
    role="web research agent",
    goal="search the web thoroughly for travel information",
    attributes="hardworking, diligent, thorough, comprehensive.",
    llm=OpenrouterModels.haiku,
    tools=[WebTools.serper_search, WikipediaTools.search_articles, WikipediaTools.search_images]
)

travel_agent = Agent(
    role="travel agent",
    goal="assist the traveller with their request",
    attributes="friendly, hardworking, and comprehensive in reporting back to users",
    llm=OpenrouterModels.haiku,
    tools=[AmadeusTools.search_flights, WebTools.serper_search, WebTools.get_weather_data]
)

# Synchronous approach
def research_destination_sync(destination, interests):
    return Task.create(
        agent=web_research_agent,
        context=f"User Destination: {destination}\nUser Interests: {interests}",
        instruction=f"Research {destination} and write a comprehensive report with images embedded in markdown."
    )

def main_sync():
    destination = input("Enter a destination: ")
    interests = input("Enter your interests: ")

    destination_report = research_destination_sync(destination, interests)
    print(destination_report)

# Asynchronous approach (for concurrent execution)
import asyncio

async def research_destination_async(destination, interests):
    return await Task.create_async(
        agent=web_research_agent,
        context=f"User Destination: {destination}\nUser Interests: {interests}",
        instruction=f"Research {destination} and write a comprehensive report with images embedded in markdown."
    )

async def main_async():
    destination = input("Enter a destination: ")
    interests = input("Enter your interests: ")

    destination_report = await research_destination_async(destination, interests)
    print(destination_report)

# Choose your approach
if __name__ == "__main__":
    # Uncomment one of these:
    # main_sync()  # For synchronous execution
    asyncio.run(main_async())  # For asynchronous execution
```

### Multi-Agent Teams

Orchestra enables the creation of powerful multi-agent teams by assigning tasks to agents equipped with specific tools. This approach facilitates complex workflows and collaborative problem-solving, allowing you to tackle intricate challenges that require diverse skills and knowledge.

In a multi-agent team, each agent is designed with a specialized role, a set of tools, and specific expertise. By combining these agents, you can create AI workflows capable of handling a wide range of tasks, from research and analysis to problem-solving and code generation.

**Acknowledgment**: Mainframe-Orchestra is a fork and further development of [TaskflowAI](https://github.com/philippe-page/taskflowai) by Philippe Pagé.
