# Conversation Tools

The ConversationTools class primarily facilitates human-in-the-loop processes within the Orchestra framework. It provides utility methods that enable human interaction, enhancing Orchestra's capabilities for creating aligned and responsive AI systems. These tools can be used whenever you need to incorporate human feedback, guidance, or oversight into your task-based workflows.

### Class Methods

##### ask_user(question: str) -> str

Facilitates human-in-the-loop processes by allowing agents to request input from human users. This method is crucial for maintaining alignment in AI systems and creating effective AI assistants that can work alongside humans, seeking clarification or guidance when needed.

```python
user_input = ConversationTools.ask_user("What specific criteria should I consider for this task?")
print(f"User's criteria: {user_input}")
```

### Usage in Tasks

Here's an example of how you might use ConversationTools to create a human-aligned agent system for task execution:

```python
from mainframe_orchestra import Agent, Task, OpenrouterModels, set_verbosity, ConversationTools, WebTools

set_verbosity(1)

research_assistant = Agent(
    role="Research Assistant",
    goal="Provide accurate and relevant information based on user requirements",
    attributes="Adaptable, detail-oriented, and responsive to user feedback",
    tools={WebTools.exa_search, ConversationTools.ask_user},
    llm=OpenrouterModels.haiku
)

def research_task(topic: str):
    initial_query = ConversationTools.ask_user(f"What specific aspects of {topic} should I focus on?")
    research_results = WebTools.exa_search(f"{topic} {initial_query}")
    
    feedback = ConversationTools.ask_user("Is this information sufficient, or should I refine my search?")
    if "refine" in feedback.lower():
        refined_query = ConversationTools.ask_user("How should I adjust my search?")
        research_results += WebTools.exa_search(refined_query)
    
    return research_results

Task(research_task, args=["artificial intelligence"])
```

### Key Concepts

1. **Human-in-the-Loop**: The `ask_user` method allows for seamless integration of human input, ensuring AI systems remain aligned with human preferences and values during task execution.

2. **Alignment and Course Correction**: By incorporating human feedback at key points, agents can adjust their approach and ensure they're meeting user expectations.

3. **Adaptive Task Execution**: The ability to seek clarification or additional guidance allows for more flexible and responsive task workflows.

### Usage Notes

1. Use human input strategically to maintain a balance between automation and human oversight in your task workflows.
2. Design your tasks with clear points for potential human intervention, allowing for course correction and alignment checks.
3. Consider using human input for critical decision points, validation of results, or when facing ambiguity in task requirements.

By leveraging these tools, Orchestra can support the creation of AI systems that are not only efficient but also closely aligned with human intent and values. This approach ensures that AI assistants remain helpful, responsive, and adaptable to user needs throughout the task execution process.