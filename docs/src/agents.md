# Agents

Agents in Orchestra are components that encapsulate specific personas to be assigned to tasks. They are designed to perform tasks in a manner consistent with their defined role, goal, and attributes. Agents are typically configured with a specific LLM and, if needed, a set of tools tailored to their role, enabling them to effectively execute their assigned tasks.

The reusability of agents in Orchestra not only streamlines workflow design but also contributes to the overall consistency of output. By utilizing the same agent across different tasks within its area of expertise, you can expect uniform behavior and response styles, which is particularly valuable in maintaining a coherent user experience or adhering to specific operational standards.

### Configuring Agent Intelligence and Capabilities

In Orchestra, agents can be further customized by setting specific LLMs and tool sets. This configuration allows you to fine-tune the agent's intelligence level, associated costs, and functional capabilities, effectively creating specialized teams of agents.

### Anatomy of an Agent

An agent in Orchestra is defined by four core components:

- **role**: Defines the agent's purpose or function within the Orchestra workflow. e.g. "Web Researcher".
- **goal**: Specifies the desired outcome or objective that the agent aims to achieve. e.g. "find relevant Python agent repositories with open issues".
- **attributes** (Optional): Additional characteristics or traits that shape the agent's behavior and personality. e.g. "analytical, detail-oriented, and determined to write thorough reports".
- **llm**: The underlying language model assigned to the agent for generating responses. e.g. `OpenrouterModels.haiku`.

### Creating an Agent

To create an agent in Orchestra, you can use the Agent class provided by the library. Here's an example:

```python
from mainframe_orchestra import Agent, OpenrouterModels

customer_support_agent = Agent(
    role="customer support representative",
    goal="to resolve customer inquiries accurately and efficiently",
    attributes="friendly, empathetic, and knowledgeable about the product",
    llm=OpenrouterModels.haiku
)
```

### Assigning Agents to Tasks

Here's an example demonstrating how an agent can be created and then integrated into multiple tasks within a Orchestra workflow:

```python
from mainframe_orchestra import Task, Agent, OpenrouterModels

data_analyst_agent = Agent(
    role="data analyst",
    goal="to provide insights and recommendations based on data analysis",
    attributes="analytical, detail-oriented, and proficient in statistical methods",
    llm=OpenrouterModels.haiku
)

def analysis_task (sales_data):
    return Task.create(
       agent=data_analyst_agent,
       context=f"The sales data for the past quarter is attached: '{sales_data}'.",
       instruction="Analyze the sales data and provide recommendations for improving revenue."
    )
```

### Assigning Tools to Agents

Agents can be assigned tools to enhance their capabilities and enable them to perform specific actions. Tools are functions that the agent can use to interact with external systems, process data, or perform specialized tasks. 

The agent will have the opportunity to use tools provided to the agent or the task to assist in its completion. The tools are passed to the agent's 'tools' parameter during initialization, and the agent will then be able to see and use the tools before completing their final response. They can call tools once, recursively, or multiple times in parallel. For more on tool use see the [agentic tool use](/agentic-tool-use) page.

Here's an example of assigning tools to an agent:

```python
from mainframe_orchestra import Agent, GitHubTools, OpenaiModels

researcher = Agent(
    role="GitHub researcher",
    goal="find relevant Python agent repositories with open issues",
    attributes="analytical, detail-oriented, able to assess repository relevance and popularity",
    llm=OpenaiModels.gpt_4o_mini,
    tools={GitHubTools.search_repositories, GitHubTools.get_repo_details}
)
```

In this example, the researcher agent is assigned two tools from the GitHubTools module. These tools allow the agent to search for repositories and get repository details, which are essential for its role as a GitHub researcher. Tools are passed to the agent's 'tools' parameter during initialization.

##### Advanced Agent Parameters

Agents can be assigned additional parameters to tune their behavior. These additional params control model temperature and max tokens. Default temperature is 0.7 and max tokens is 4000. You can set temperature and max tokens in the agent definition and they will override the defaults set in the llm. Here's an example:

```python
from mainframe_orchestra import Agent, OpenrouterModels

assistant_agent = Agent(
    role="assistant",
    goal="to provide insights and recommendations based on data analysis",
    llm=OpenrouterModels.haiku,
    max_tokens=500,
    temperature=0.5
)
```

These additional settings are optional and are often not required unless custom or specific temperature and max tokens are required. The default temperature of 0.7 and max tokens of 4000 covers most use cases, but programming or long responses may benefit from custom temperature and max tokens.

##### Prompting

Prompting involves crafting effective prompts for agent roles, goals, and attributes to elicit desired behaviors and responses from the language model. Here are some tips for effective prompting:

- Use clear and concise language that captures the essence of the agent's role and goal.
- Use the optional attributes field to provide additional behavioral cues and suggestions based on feedback from tests.
- Experiment with different prompt variations and evaluate their impact on agent performance.
- Use the attributes field to provide additional behavioral cues and suggestions based on feedback from tests".

Testing and iterative development is key to creating effective prompts. The feedback from the initial runs will be used to refine the prompts and improve the performance of the agents. It's worth testing and adjusting early in the process as you develop out your multi-agent team or task flows.

By incorporating these advanced techniques, you can create agents that can handle complex tasks, adapt to user preferences, and provide more personalized and context-aware responses.
