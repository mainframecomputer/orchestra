[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/mainframecomputer/orchestra/issues)
[![PyPI version](https://badge.fury.io/py/mainframe-orchestra.svg)](https://pypi.org/project/mainframe-orchestra/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Twitter](https://img.shields.io/twitter/follow/orchestraorg?label=Follow%20@orchestraorg&style=social)](https://twitter.com/orchestraorg)

# Orchestra

Cognitive Architectures for Multi-Agent Teams.

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
GPT-4o, GPT-4o Mini, & more

### Anthropic
Claude 3 Haiku, Claude 3 Sonnet, Claude 3 Opus, Claude 3.5 Sonnet, & more

### Openrouter
GPT-4 Turbo, Claude 3 Opus, Mixtral 8x7B, Llama 3.1 405B, & more

### Ollama
Mistral, Mixtral, Llama 3.1, Qwen, Gemma, & more

### Groq
Mixtral 8x7B, Llama 3, Llama 3.1, Gemma, & more

Each provider is accessible through a dedicated class (e.g., `OpenaiModels`, `AnthropicModels`, etc.) with methods corresponding to specific models. This structure allows for painless switching between models and providers, enabling users to leverage the most suitable LLM for their tasks.

## Tools

Mainframe-Orchestra comes with a set of built-in tools that provide a wide range of functionalities, skills, actions, and knowledge for your agents to use in their task completion.  

- WebTools: For web scraping, searches, and data retrieval with Serper, Exa, WeatherAPI, etc.
- FileTools: Handling various file operations like reading CSV, JSON, and XML files.
- GitHubTools: Interacting with GitHub repositories, including listing contributors and fetching repository contents.
- CalculatorTools: Performing date and time calculations.
- EmbeddingsTools: Generating embeddings for text.
- WikipediaTools: Searching and retrieving information from Wikipedia.
- AmadeusTools: Searching for flight information.
- LangchainTools: A wrapper for integrating Langchain tools to allow agents to use tools in the Langchain catalog.
- Custom Tools: You can also create your own custom tools to add any functionality you need.

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
    attributes="You have expertise in orchestrating the orchestra.",
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

By combining agents, tasks, tools, and language models, you can create a wide range of workflows, from simple pipelines to complex multi-agent teams.

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
