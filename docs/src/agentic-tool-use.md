# Tool Use

Orchestra gives agents the ability to use tools dynamically, enhancing their problem-solving capabilities. This section explores the mechanism behind tool use, how tools are assigned to agents, and best practices for optimizing tool-based tasks.

### The Tool Use Loop

The tool use loop is a core feature of Orchestra that allows agents to dynamically select and use tools to complete tasks. When tools are assigned to a task, the execution enters this loop, enabling the LLM to make decisions about tool usage based on the current state of the task.

When a task is run, it first checks if tools are assigned to the task (either through the agent or directly to the task). If tools are present, it enters the tool-use loop by calling `_execute_tool_loop()`. If no tools are assigned, it skips the loop and directly executes the task using the LLM.

The method now also accepts an optional `callback` parameter, allowing for real-time updates during task execution. 

### Tool Loop Mechanism

The tool use loop consists of several key steps:

- **Task Analysis:** The LLM evaluates the current state of the task, including the original instruction, context, and if available, the results of previous tool usage. 
- **Decision Making:** Based on the analysis, the LLM decides whether to use a tool or if it has sufficient information to complete the task. 
- **Tool Selection and Parameter Specification:** If tool use is necessary, the LLM selects the most appropriate tool(s) and specifies the parameters in a JSON format. 
- **Tool Execution:** The selected tool(s) are executed with the specified parameters, and the results are captured. 
- **Context Update:** The tool execution results are added to the task's context for consideration in subsequent iterations. 
- **Loop Continuation:** The process repeats until the LLM determines it has gathered enough information or reaches a maximum number of iterations.

The LLM communicates its decisions through a JSON response format:

```
{
    "tool_calls": [
        {
            "tool": "tool_name",
            "params": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    ]
}
```

Or, when no further tool calls are needed:

```
{
    "status": "READY"
}
```

### Tool Assignment and Usage

Tools are assigned to a task by passing a set of callable objects to the `tools` parameter when creating a Task object. The LLM can then use these tools as needed during task execution.

```
from mainframe_orchestra import Agent, Task, WebTools

web_researcher_agent = Agent(
    role="Research Assistant",
    goal="answer user queries",
    llm=llm,
    tools={WebTools.exa_search, WebTools.get_weather_data}
)
```

In this example, two tools (`exa_search` and `get_weather_data`) are assigned to the agent. The LLM can choose to use these tools as needed to answer the user's query.

### Retry Loop and Error Handling

Orchestra implements a robust retry loop mechanism that works in conjunction with comprehensive error messaging from tools. This feature is crucial for handling cases of malformed input, improper tool usage, or correctable mistakes made by the LLM.

The retry loop begins when a tool encounters an error, such as invalid parameters or logical inconsistencies, and returns a detailed error message. This error message is then added to the task's context, allowing the LLM to analyze and retry based on the mistake. Upon reviewing the updated context, including the error message, the LLM attempts to correct its approach. It may adjust its tool selection, modify parameters, or change its strategy entirely based on the error information.

This process continues iteratively, with the LLM having the ability to correct errors and adjust its approach, leading to more accurate and effective tool use. The retry loop is particularly effective when tools provide specific, actionable error messages, enabling the LLM to make informed decisions and corrections.

### Maximum Iterations and Issue Prevention

To prevent potential issues arising from infinite loops or excessive token consumption, Orchestra implements a maximum iterations limit in its tool use loop. This safeguard ensures that the task execution process remains bounded and controlled, even in complex scenarios.

The maximum iterations limit acts as a safety net, preventing the LLM from getting stuck in a loop of repeated tool calls or failing to reach a conclusion. When the limit is reached, the system gracefully terminates the tool use loop and proceeds to the final task execution phase. This approach ensures that the task always completes within a reasonable timeframe and resource allocation. 

It's important to note that tasks completed without successfully using the tools should be investigated if the maximum iterations is reached and the `Warning: Maximum iterations of tool use loop reached without completion.` is printed. There are many potential reasons for this type of outcome, including the model being too small or if the tools are returnign unexpected values. To debug this, you can call the tool directly to inspect its outputs, and try to work with different LLMs to test. The docstrings for each tool should also provide guidance on the expected format of the tool's output.

### Ensuring Tool Functionality

To be used as a tool in Orchestra, a Python function should generally meet the following requirements:

- Accept keyword arguments for flexibility in parameter specification.
- Return data that can be converted to a string representation.
- Be self-contained.
- Include a clear docstring explaining its purpose, parameters, and return value.
- Returns error messages specific to error cases where the agent can self-correct in the retry loop.

For more details on how to create your own tools, see the [Custom Tools Page](./tools/writing-custom-tools).

### Guiding LLM Tool Usage

Assign tools that complement each other and can be used in combination to solve complex problems.

Strike a balance between task-specific tools and more general-purpose tools. This approach enables the LLM to handle both anticipated and unexpected requirements efficiently.

While the LLM has autonomy in tool usage, you can provide guidance to improve its decision-making process.

##### Clear Tool Documentation

Ensure each tool function has a clear, concise docstring explaining its purpose, parameters, and return value. The LLM uses this information to make informed decisions about tool usage.

##### Contextual Hints

Provide hints or guidelines in the task instruction or context about when certain tools might be useful. This can steer the LLM towards effective tool usage without overly constraining its decision-making.

```python
def response_task(agent, user_query):
    return Task.create(
        agent=agent,
        context=f"Customer inquiry to resolve: {user_query}",
        instruction="""
        Answer the customer's question using the available tools. 
        If the query is about order status, use the check_order_status tool. 
        For complex issues not covered by specific tools, search the knowledge base. 
        If you can't find a satisfactory answer, escalate to a human agent.
        """,
    )
```

Orchestra supports tool usage patterns that can enhance the effectiveness of tasks:

##### Sequential Tool Calls

The LLM can make a series of tool calls in a specific sequence, where the output of one tool informs the input to another. This allows for complex, multi-step workflows.

##### Parallel Tool Calls

Multiple tools can be called simultaneously to gather different pieces of information in parallel, reducing the overall execution time and number of iterations required to complete the task.

##### Recursive Tool Use

The LLM can use tools recursively, calling a tool multiple times with refined parameters based on previous results. This enables iterative problem-solving approaches.

### Conclusion

Agentic tool use in Orchestra provides a powerful mechanism for creating flexible, intelligent workflows. By carefully designing and assigning tools, and providing appropriate guidance, you can enable LLMs to tackle complex tasks efficiently and effectively. The updated implementation offers more control over task execution, better error handling, and real-time feedback through callbacks.
