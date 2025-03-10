[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/mainframecomputer/orchestra/issues)
[![PyPI version](https://badge.fury.io/py/mainframe-orchestra.svg)](https://pypi.org/project/mainframe-orchestra/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter](https://img.shields.io/twitter/follow/orchestraorg?label=Follow%20@orchestraorg&style=social)](https://twitter.com/orchestraorg)

# Orchestra

Cognitive Architectures for Multi-Agent Teams.

## Table of Contents
- [Orchestra](#orchestra)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Core Components](#core-components)
  - [Supported Language Models and Providers](#supported-language-models-and-providers)
    - [OpenAI](#openai)
    - [Anthropic](#anthropic)
    - [Openrouter](#openrouter)
    - [Ollama](#ollama)
    - [Groq](#groq)
    - [TogetherAI](#togetherai)
    - [Gemini](#gemini)
    - [Deepseek](#deepseek)
  - [Tools](#tools)
    - [Built-in Tools](#built-in-tools)
      - [Data \& File Operations](#data--file-operations)
      - [Web \& API Integration](#web--api-integration)
      - [Financial \& Data Analysis](#financial--data-analysis)
      - [Media \& Content](#media--content)
      - [Integration Tools](#integration-tools)
    - [Custom Tools](#custom-tools)
  - [Multi-Agent Teams](#multi-agent-teams)
  - [Conduct and Compose](#conduct-and-compose)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Support](#support)

## Overview

Mainframe-Orchestra is a lightweight, open-source agentic framework for building LLM-based pipelines and multi-agent teams. It implements a unique approach to agent orchestration that goes beyond simple routing, enabling complex workflows.

## Key Features

- **Modularity**: Modular architecture for easy building, extension, and integration
- **Agent Orchestration**: Agents can act as both executors and conductors, enabling dynamic task decomposition and coordination among agents
- **Phased Task Execution**: Reduces cognitive load on LLMs through structured thinking patterns
- **Tool Integration**: Simple docstring-based tool definitions without complex JSON schemas
- **Streaming Support**: Real-time output streaming with both sync and async support
- **Built-in Fallbacks**: Graceful handling of LLM failures with configurable fallback chains

## Installation

Install Orchestra using pip:

```bash
pip install mainframe-orchestra
```

## Quick Start

Here's a simple example to get you started:

```python
from mainframe_orchestra import Agent, Task, OpenaiModels, WebTools, set_verbosity

set_verbosity(1)

research_agent = Agent(
    agent_id="research_assistant_1",
    role="research assistant",
    goal="answer user queries",
    llm=OpenaiModels.gpt_4o,
    tools={WebTools.exa_search}
)

def research_task(topic):
    return Task.create(
        agent=research_agent,
        instruction=f"Use your exa search tool to research {topic} and explain it in a way that is easy to understand.",
    )

result = research_task("quantum computing")
print(result)
```

## Core Components

**Tasks**: Discrete units of work

**Agents**: Personas that perform tasks and can be assigned tools

**Tools**: Wrappers around external services or specific functionalities

**Language Model Interfaces**: Consistent interface for various LLM providers

## Supported Language Models and Providers

Orchestra supports a wide range of language models from a number of providers:

### OpenAI
GPT-4.5-preview, GPT-4o, GPT-4o Mini, & Custom defined models

Orchestra supports customizing the OpenAI base URL, allowing you to connect to OpenAI-compatible APIs or proxies:

```python
# Method 1: Set via environment variable
import os
os.environ["OPENAI_BASE_URL"] = "https://your-custom-endpoint.com/v1"

# Method 2: Set globally for all OpenAI requests
from mainframe_orchestra.llm import OpenaiModels
OpenaiModels.set_base_url("https://your-custom-endpoint.com/v1")

# Method 3: Set for a specific request
response, error = await OpenaiModels.gpt_4o(
    messages=[{"role": "user", "content": "Hello"}],
    base_url="https://your-custom-endpoint.com/v1"
)
```

### Anthropic
Claude 3 Haiku, Claude 3 Sonnet, Claude 3 Opus, Claude 3.5 Sonnet, Claude 3.7 Sonnet, & Custom defined models

### Openrouter
GPT-4 Turbo, Claude 3 Opus, Mixtral 8x7B, Llama 3.1 405B, & Custom defined models

### Ollama
Mistral, Mixtral, Llama 3.1, Qwen, Gemma, & Custom defined models

### Groq
Mixtral 8x7B, Llama 3, Llama 3.1, Gemma, & Custom defined models

### TogetherAI
Custom defined models

### Gemini
Gemini 1.5 Flash, Gemini 1.5 Flash 8B, Gemini 1.5 Pro, & Custom defined models

### Deepseek
Deepseek Reasoner, Deepseek Chat, & Custom defined models

Each provider is accessible through a dedicated class (e.g., `OpenaiModels`, `AnthropicModels`, etc.) with methods corresponding to specific models. This structure allows for painless switching between models and providers, enabling users to leverage the most suitable LLM for their tasks.

## Tools

Mainframe-Orchestra comes with a comprehensive set of built-in tools that provide various functionalities for your agents. Here's an overview of the available tool categories:

### Built-in Tools

#### Data & File Operations
- **FileTools**: Read and write CSV, JSON, XML, and other file formats
- **TextSplitters**: Tools for chunking and splitting text documents
- **EmbeddingsTools**: Generate embeddings for text content
- **FaissTools**: Vector storage and similarity search operations
- **PineconeTools**: Vector database operations with Pinecone

#### Web & API Integration
- **WebTools**: Web scraping, searches, and data retrieval (Serper, Exa, etc.)
- **WikipediaTools**: Search and retrieve Wikipedia content
- **AmadeusTools**: Flight information and travel data
- **GitHubTools**: GitHub repository operations and content access
- **LinearTools**: Linear API-based tools for creating, updating, and retrieving tasks

#### Financial & Data Analysis
- **YahooFinanceTools**: Stock market data and financial analysis
- **FredTools**: Federal Reserve Economic Data access
- **CalculatorTools**: Date, time, and mathematical calculations
- **MatplotlibTools**: Data visualization and plotting

#### Media & Content
- **AudioTools**: Audio processing and manipulation
- **TextToSpeechTools**: Text-to-speech conversion using ElevenLabs and OpenAI APIs
- **WhisperTools**: Audio transcription and translation using OpenAI's Whisper API

#### Integration Tools
- **LangchainTools**: Wrapper for accessing the Langchain tools ecosystem

### Custom Tools

Mainframe-Orchestra supports creating custom tools to extend functionality beyond the built-in tools. Custom tools can be implemented either as static methods or as class instance methods for more complex operations. Here's a basic example:

```python
import numpy as np
from typing import List, Union

class NumpyTools:
    @staticmethod
    def array_mean(arr: Union[List[float], np.ndarray]) -> Union[float, str]:
        """
        Calculate the mean of a given array.

        Args:
            arr (Union[List[float], np.ndarray]): Input array or list of numbers.

        Returns:
            Union[float, str]: The mean of the input array as a float, or an error message as a string.
        """
        try:
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return "Error: Input array is empty."
            return float(np.mean(arr))
        except TypeError as e:
            return f"Error: Invalid input type. Expected a list or numpy array of numbers. Details: {e}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {e}"
```

Tools can be assigned to agents during initialization:

```python
agent = Agent(
    agent_id="my_agent",
    tools={NumpyTools.array_mean, WebTools.exa_search}
)
```

For detailed documentation on creating custom tools, including best practices for error handling and API integration, visit our [Custom Tools Documentation](https://docs.orchestra.org/orchestra/tools/writing-custom-tools).

## Multi-Agent Teams

Mainframe-Orchestra allows you to create multi-agent teams that can use tools to complete a series of tasks. Here's an example of a finance agent that uses multiple agents to analyze a stock:

```python
from mainframe_orchestra import Task, Agent, Conduct, OpenaiModels, WebTools, YahooFinanceTools

# Create specialized agents
market_analyst = Agent(
    agent_id="market_analyst",
    role="Market Microstructure Analyst",
    goal="Analyze market microstructure and identify trading opportunities",
    attributes="You have expertise in market microstructure, order flow analysis, and high-frequency data.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.calculate_returns, YahooFinanceTools.get_historical_data}
)

fundamental_analyst = Agent(
    agent_id="fundamental_analyst",
    role="Fundamental Analyst",
    goal="Analyze company financials and assess intrinsic value",
    attributes="You have expertise in financial statement analysis, valuation models, and industry analysis.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_financials, YahooFinanceTools.get_ticker_info}
)

technical_analyst = Agent(
    agent_id="technical_analyst",
    role="Technical Analyst",
    goal="Analyze price charts and identify trading patterns",
    attributes="You have expertise in technical analysis, chart patterns, and technical indicators.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_historical_data}
)

sentiment_analyst = Agent(
    agent_id="sentiment_analyst",
    role="Sentiment Analyst",
    goal="Analyze market sentiment, analyst recommendations and news trends",
    attributes="You have expertise in market sentiment analysis.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_recommendations, WebTools.serper_search}
)

conductor_agent = Agent(
    agent_id="conductor_agent",
    role="Conductor",
    goal="Conduct the orchestra",
    attributes="You have expertise in orchestrating the agents in your team.",
    llm=OpenaiModels.gpt_4o,
    tools=[Conduct.conduct_tool(market_analyst, fundamental_analyst, technical_analyst, sentiment_analyst)]
)

def chat_task(conversation_history, userinput):
    return Task.create(
        agent=conductor_agent,
        messages=conversation_history,
        instruction=userinput
    )

def main():
    conversation_history = []
    while True:
        userinput = input("You: ")
        conversation_history.append({"role": "user", "content": userinput})
        response = chat_task(conversation_history, userinput)
        conversation_history.append({"role": "assistant", "content": response})
        print(f"Market Analyst: {response}")

if __name__ == "__main__":
    main()
```

Note: this example requires the yahoofinance and yfinance packages to be installed. You can install them with `pip install yahoofinance yfinance`.

## Conduct and Compose
The `Conduct` and `Compose` tools are used to orchestrate and compose agents. Conduct is used to actually instruct and orchestrate a team of agents, while Compose is used in addition to the Conduct tool to enrich the orchestration process with additional complexity as a preprocessing step. It's important to note that Conduct is required for the orchestration process to work, while Compose is an optional additional tool that can be used to enrich the orchestration process.

By combining agents, tasks, tools, and language models, you can create a wide range of workflows, from simple pipelines to complex multi-agent teams.

## MCP Integration
- **MCPOrchestra**: Adapter for integrating with Model Context Protocol (MCP) servers, allowing agents to use any MCP-compatible toolkits / servers
  - Connect to FastMCP, Playwright, Slack, Filesystem, and other MCP-compatible servers
  - List available tools from an MCP server
  - Convert external tools into Orchestra-compatible callables for agents to use

For documentation on MCP integration, visit our [MCP Integration Guide](https://docs.orchestra.org/orchestra/mcp-integration-with-orchestra).


## Streaming Support

Orchestra supports streaming of LLM responses. When using streaming, you need to use an async approach:

```python
import asyncio
from mainframe_orchestra import Agent, Task, OpenaiModels, WebTools, set_verbosity

set_verbosity(1)

research_agent = Agent(
    agent_id="research_assistant_1",
    role="research assistant",
    goal="answer user queries",
    llm=OpenaiModels.gpt_4o,
    tools={WebTools.exa_search}
)

async def research_task_streaming():
    # Create the task and await it
    task = await Task.create(
        agent=research_agent,
        instruction="Use your exa search tool to research quantum computing and explain it in a way that is easy to understand.",
        stream=True
    )

    # Process the streaming output
    async for chunk in task:
        print(chunk, end="", flush=True)
    print()  # Add a newline at the end

# Run the async function
if __name__ == "__main__":
    asyncio.run(research_task_streaming())
```

The key points for streaming:
1. Make your function async
2. Set `stream=True` in the Task.create call
3. Await the Task.create() call to get the streaming task
4. Use `async for` to process the streaming chunks
5. Run the async function with asyncio.run()


## Documentation

For more detailed information, tutorials, and advanced usage, visit our [documentation](https://docs.orchestra.org).

## Contributing

Mainframe-Orchestra depends on and welcomes community contributions! Please review contribution guidelines and submit a pull request if you'd like to contribute.

## License

Mainframe-Orchestra is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Orchestra is a fork and further development of [TaskflowAI](https://github.com/philippe-page/taskflowai).

## Support

For issues or questions, please file an issue on our [GitHub repository issues page](https://github.com/mainframecomputer/orchestra/issues).

⭐️ If you find Mainframe-Orchestra helpful, consider giving it a star!

Happy building!
