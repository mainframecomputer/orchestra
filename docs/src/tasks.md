# Tasks

The Task class is the fundamental building block of Orchestra, encapsulating the concept of a single discrete unit of work within an AI workflow. 

### Task-Centric Approach

Within organizations, businesses, operations, and projects, tasks are the fundamental units of activity. These tasks, performed by individuals with specific roles and utilizing various tools, form the backbone of standard operating procedures (SOPs). 

Many AI frameworks are built around conversation patterns and inter-agent communication, and while this approach has its merits, it often lacks the structure and predictability that real-world operations require. 

The task-centric design of Orchestra closely mirrors how human organizations function. By focusing on discrete units of work with clear inputs and outputs, the framework provides a structured environment that improves reliability, consistency, and manageability of AI systems.

The task-centric design of Orchestra also contributes to the scalability of AI systems. New tasks can be added or existing ones modified without disrupting the entire workflow, allowing organizations to expand and refine their AI-driven processes incrementally.

This page covers the Task class, its implementation, and best practices for utilizing it effectively in your Orchestra projects.

### Structure and Attributes

The Task class in Orchestra is defined in the task.py file. It's the core component for creating and executing tasks.

The `Task.create()` method is used to create a task and most often is used with the following parameters:
- agent: The agent assigned to the task
- context: Additional context for the task
- instruction: Specific directions for the task

##### Additional Parameters

The task class has a few additional optional parameters that can be used to further customize the task.

- `temperature`: The temperature of the language model.
- `max_tokens`: The maximum number of tokens to generate.
- `tools`: A list of tools that the language model / agent can use, specific to the task. Adds to the agents tools.
- `llm`: The language model to use, specific to the task. Overrides the default language model for the agent assigned to the task.
- `require_json_output`: If provided and set to `True`, the task will expect a valid JSON output from the language model. It utilizes JSON mode from the LLM API providers to ensure the output is valid JSON.
- `stream`: Set to `True` to enable streaming of the final LLM response.
- `initial_response`: Set to `True` to provide an initial response before tool execution.
- `tool_summaries`: Set to `True` to include explanatory summaries for tool calls.

### Execution and Integration

The `Task.create()` method is responsible for the creation and execution of a task, generating the response from the specified language model. It handles prompt construction, language model processing, tool-use and function calling execution, error handling, and retries.

Here's an example of how to use the Task class, where the task is to research events in a given location for a given date span, and write a report on the events found.

```python
def research_events(destination, dates, interests):
    events_report = Task.create(
        agent=web_research_agent,
        context=f"User's intended destination: {destination}\n\nUser's intended dates of travel: {dates}\n\nUser Interests: {interests}",
        instruction="Use your tools to research events in the given location for the given date span. Your final response should be a comprehensive report on events in the area for that time period."
    )
    return events_report
```

### Streaming Responses

To stream the output of a task, you can set the flag to true and use the `Task.process_stream()` method. This method takes the output of the `Task.create()` method and processes it.

```python
streaming_output = Task.create(
    agent=agent,
    instruction="What is 157 + 42? What is the answer of that - 32? Write your answer in a poem.",
    stream=True
)
Task.process_stream(streaming_output)
```

### Initial Responses

Initial responses enable the LLM to provide a preliminary answer before executing any tools, giving users immediate feedback while more detailed processing occurs in the background. This is especially useful in conversational contexts.

```python
output = Task.create(
    agent=agent,
    instruction="What is 157 + 42? What is the answer of that - 32? Write your answer in a poem.",
    initial_response=True
)
print(output)
```

### Tool Summaries

Tool summaries provide explanatory context for each tool call, helping users understand why specific tools were used and what they're intended to accomplish. This transparency is valuable for debugging, auditing, and understanding the LLM's decision-making process.

```python
output = Task.create(
    agent=agent,
    instruction="What is 157 + 42? What is the answer of that - 32? Write your answer in a poem.",
    tool_summaries=True
)
print(output)
```

### Logging

You can enable logging for tasks by calling the `configure_logging()` method. This will log all events to a file and console. You can provide a file name to log to in the `log_file` parameter. To log to file, import the `configure_logging()` method and set it up at the beginning of your script.
```python
# Configure logging with DEBUG level and file output
configure_logging(
    level="DEBUG",
    log_file="agent_execution.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Overall Task Considerations

### Task Decomposition

Task decomposition is a crucial strategy in Orchestra for efficiently managing complex problems. This process involves breaking down large, intricate tasks into smaller, more manageable subtasks, thereby enhancing the overall performance and reliability of AI workflows. The approach begins with a comprehensive analysis of the problem, identifying key components or steps necessary for its resolution. This initial assessment serves as the foundation for creating well-defined subtasks, each with a specific objective.

Once the problem is dissected, individual subtasks are created as discrete units of work contributing to the overall solution. Each subtask should have a focused purpose. This granularity allows for better resource allocation. After defining subtasks, it's crucial to identify and map out input and output flows between them, helping organize the workflow and determine the ideal execution sequence.

Task decomposition not only makes complex problems more manageable but also enhances the flexibility and maintainability of AI workflows. It facilitates easier debugging, as issues can be isolated to specific subtasks. By mastering task decomposition, you can significantly improve the effectiveness and scalability of your Orchestra implementations, creating more robust and effective task flows.

### Managing Context Window Attention

The concept of context window optimization in Orchestra's Task class aligns with recent findings on large language model (LLM) attention mechanisms. As described in the research paper "Lost in the Middle: How Language Models Use Long Contexts" by Liu et al. (2023), LLMs exhibit a U-shaped attention pattern, with higher focus on information at the beginning and end of the input context.

Orchestra leverages this behavior by structuring the input strategically: placing critical elements like agent role and goals at the start, broader context in the middle, and specific instructions at the end. This approach aims to balance strict guideline adherence with contextual flexibility, potentially enhancing task execution effectiveness. The structure mirrors the "Serial Position Effect" observed in human cognition, where items at the beginning and end of a sequence are better remembered.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Belinkov, Y., Liang, P., & Hashimoto, T. B. (2023). Lost in the Middle: How Language Models Use Long Contexts. [Read the full paper here.](https://arxiv.org/pdf/2307.03172)

### Managing Coherence in Task Execution

The management in Orchestra's Task class addresses a critical challenge in language model performance known as "Coherence Loss" or "Coherence Collapse." This phenomenon, particularly relevant for smaller or less expensive models and in scenarios involving extensive context, refers to a state where the model's output becomes repetitive, nonsensical, or loses logical flow.

Orchestra implements several strategies to mitigate coherence loss. The system tracks tool call repetitions by generating a hash from the tool name and parameters, allowing it to detect when the language model is stuck in a loop. Upon the first repetition, the system issues a warning to the model, encouraging it to adjust parameters or proceed with the final response. If a second repetition occurs, the system forcibly exits the loop, preventing further resource waste and ensuring task progression.

This approach aligns with findings from recent research on language model behavior, such as the work by Zhang et al. (2023) on "Losing Coherence: Why Larger Language Models are Prone to Hallucination." Their study highlights how models can lose track of context in extended generations, leading to inconsistent or irrelevant outputs. Orchestra's coherence management techniques aim to address these issues by providing structured guidance and implementing fail-safes to maintain task focus and output quality.

By incorporating these coherence management strategies, Orchestra enhances the reliability and efficiency of AI-driven workflows, particularly in complex scenarios or when working with models of varying capabilities. This proactive approach to maintaining coherence contributes to the overall robustness of task execution within the framework.