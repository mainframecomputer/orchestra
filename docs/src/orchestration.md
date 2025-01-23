# Orchestration

Orchestra's orchestration module represents an approach to agent collaboration and task delegation that goes beyond simple message routing. Orchestra enables dynamic task decomposition, agent assignment, and data flow management through a tool-based architecture. This approach allows agents to not only perform tasks but also coordinate complex workflows, creating a dynamic and adaptive system.

## The Orchestration Module

At the heart of Orchestra's capabilities is the `Conduct` class, which provides a mechanism for enabling agents to delegate tasks to other agents through a tool-based architecture. The delegation capability is implemented as a tool that can be granted to any agent, allowing for flexible and dynamic orchestration patterns. This means that any agent with the conduct tool can coordinate and delegate tasks to other agents, creating nested orchestration patterns without relying on a central orchestrator.

The `Conduct` and `Compose` tools are used to orchestrate and compose agents. Conduct is used to actually instruct and orchestrate a team of agents, while Compose is used in addition to the Conduct tool to enrich the orchestration process with additional complexity as a preprocessing step. It's important to note that Conduct is required for the orchestration process to work, while Compose is an optional additional tool that can be used to enrich the orchestration process.

### The Conduct Tool

The `Conduct` tool serves as a factory, creating a specialized tool function that encapsulates the delegation logic. When added to an agent, this tool provides the ability to coordinate and delegate tasks to a set of eligible agents that were registered during the tool's creation. The implementation carefully manages the flow of information between agents, tracking dependencies, and ensuring proper message passing through event queues.

```python
class TaskInstruction(BaseModel):
    task_id: str
    agent_id: str
    instruction: str
    use_output_from: List[str] = []
```

A key feature of the orchestration system is its handling of task dependencies through the `TaskInstruction` model. This model enforces a structure where each task must specify its `task_id`, `agent_id`, and `instruction`, along with any dependencies on other tasks through the `use_output_from` field. This explicit dependency management ensures that agents receive necessary context from previous task results when executing their assigned tasks.

The delegation process involves careful error handling, iteration limits, and message standardization. Each delegated task creates its own message context, allowing agents to maintain clear conversation histories while referencing results from dependent tasks. The nested callback system ensures that all events—whether they're tool calls, results, or delegation messages—are properly tracked and forwarded to the appropriate handlers.

By implementing delegation as a tool, Orchestra allows for dynamic and hierarchical delegation patterns. An agent with the conduct tool can delegate tasks to other agents, who themselves might have the conduct tool and could further delegate subtasks. This creates the possibility for complex, nested orchestration patterns while maintaining clear boundaries and dependencies.

## The Compose Tool

The `Compose` tool represents an optional planning layer that works in harmony with the core `Conduct` tool, embodying the principle of separation of concerns in multi-agent orchestration. While the Conduct tool is essential and sufficient for task delegation and orchestration, Compose provides an additional layer of sophistication for complex planning scenarios. This means that agents can perform delegation and orchestration with just the Conduct tool, but can optionally leverage the Compose tool when more elaborate planning is beneficial.

When an agent possesses both compose and conduct tools, it can exercise judgment about when to employ each capability. For simple tasks, the agent might proceed directly to delegation through the conduct tool. However, when faced with complex objectives that require multiple agents and coordinated efforts, the agent can first utilize the compose tool to create a strategic plan.

The compose tool treats the planning phase as a distinct cognitive task. When creating a composition, it considers the sequence of operations and the flow of information between tasks. For instance, in a complex research task, the composition might outline a series of operations: financial data collection, web research, code analysis, and news gathering, followed by a synthesis phase that integrates these diverse information streams. This planning phase explicitly maps out data dependencies and information flows, ensuring that agents receive necessary context from previous task results.

The relationship between the compose and conduct tools creates a natural workflow where planning feeds into execution. Once a composition is created, it provides a structured blueprint that the conduct tool uses to execute the plan efficiently. This separation allows the planning phase to focus on high-level strategy and optimal task organization without getting bogged down in execution details.

By separating the delegatory planning phase (compose) from tactical execution instruction (conduct), Orchestra enables more thoughtful and efficient multi-agent operations while maintaining the flexibility to handle both simple and complex tasks appropriately. The system can scale its complexity based on the task at hand, demonstrating versatility and practical utility in real-world applications.

## Dynamic Task Decomposition and Agent Assignment

Orchestra's orchestration system allows agents to break down complex problems into smaller, manageable tasks on the fly. Rather than following rigid, predefined paths, agents can create new tasks as needed based on the evolving context of the problem. This dynamic task decomposition enables the system to adapt to new challenges and intricate problem domains.

The system doesn't just route messages to the ideal agent but makes informed decisions about which agent is best suited for each decomposed task, considering their capabilities and the overall workflow context. This level of agent assignment ensures that tasks are handled by agents with the appropriate expertise, leading to more efficient and effective problem-solving.

## Data Flows and Custom Instruction Messages

An agent with the conduct tool writes specific and tailored instructions for each agent instead of just routing the original message. This allows for custom, flexible, and tuned instruction messages that are appropriate for each subtask and agent. By crafting specific instructions, the orchestrating agent ensures that each agent has the necessary context and guidance to perform its task effectively.

The conductor or orchestrator can also chain together the tasks so that downstream tasks can actually take in the entire output of a previous task as context, so as to avoid the orchestrator needing to get the intermediate results and instructing the next agent with the data provided. This means that it avoids becoming the bottleneck, because in a task like "get a month of NVDA price data and plot it in a chart", it would have to receive the data from the first agent, and then provide it to the second in its instruction. So we handle that by actually allowing the conductor to chain tasks and determine which downstream ones depend on upstream tasks, and allow that data to flow from one to the next via this `context` param. This allows richer data flows with fewer bottlenecks, and puts less strain on the orchestrator and its need to regurgitate upstream outputs back to downstream agents. In other words, it gets out of the way of the agents and lets them do their thing.

## Beyond Routing: Agent Orchestration

Most agent frameworks today focus primarily on routing: directing messages to the appropriate agent. While often labeled as "delegation," this approach just creates paths for messages to flow between agents in predefined patterns, typically through a supervisor or static manager, and the message itself is unchanged.

Orchestra takes a fundamentally different approach in its implementation of orchestration, actively composing and coordinating complex workflows. Agents can communicate bidirectionally, share context, and collaboratively work on complex problems. They can spawn subtasks, coordinate responses, and maintain both local and shared state, enabling emergent behavior and finer-tuned control over task execution.

This architecture allows for the creation of complex multi-agent hierarchies and organizational structures. Organizations can be modeled with department-level orchestrators managing their specific agent teams, while higher-level orchestrators coordinate across departments. This modularity allows for the creation of specialized orchestration patterns tailored to specific domains or organizational needs.

## Example

Consider an example where various specialized agents are defined, each with specific roles, goals, attributes, and tools. These agents might include a Browser Agent for web searching, a GitHub Agent for repository information, an Email Agent for handling email operations, a Scheduling Agent for managing calendar events, and so on.

Here is an example of defining agents:

```python
# Define Browser Agent
browser_agent = Agent(
    agent_id="browser_agent",
    role="Browser Agent",
    goal="To help with browsing the web",
    attributes="You know you can view URLs, search specific sites, search news articles, shopping results, search images, search videos, search places, and search using the Google Lens and an image URL.",
    tools=[
        WebTools.scrape_url_with_serper,
        SearchTools.search_site,
        SearchTools.search_news,
        # Additional tools...
    ],
    llm=default_agent_llm
)

# Define GitHub Agent
github_agent = Agent(
    agent_id="github_agent",
    role="GitHub Agent",
    goal="To assist with GitHub-related tasks and repository information",
    attributes="You can assist with GitHub-related tasks, like searching repositories, getting repository directory structures, and file contents.",
    tools=[
        GitHubTools.search_repositories,
        GitHubTools.get_directory_structure,
        # Additional tools...
    ],
    llm=default_agent_llm
)

# Define Coordinator
coordinator = Agent(
    agent_id="coordinator",
    role="Coordinator",
    goal="To chat with and help the human user by coordinating your team of agents to carry out tasks",
    attributes="You know that you can delegate tasks to your team of agents, and you can take outputs of agents and use them for subsequent tasks if needed. Your team includes a Browser Agent, a GitHub Agent, an Email Agent, and others.",
    tools=[Conduct.conduct_tool(
        browser_agent,
        github_agent,
        # Additional agents...
    )],
    llm=default_coordinator_llm
)
```

In this example, the coordinator agent has the ability to delegate tasks to other agents using the conduct tool. The agents can communicate and pass information between each other, enabling complex tasks to be decomposed and executed efficiently.

## Conclusion

Orchestra's orchestration module offers a flexible approach to agent collaboration and task delegation. By treating orchestration capabilities as modular tools that can be assigned to any agent, Orchestra enables the construction of  multi-agent interactions with clear boundaries and dependencies between tasks. This tool-based architecture allows for dynamic task decomposition, intelligent agent assignment, and efficient data flows, creating a truly dynamic and adaptive system capable of handling complex, real-world applications.
