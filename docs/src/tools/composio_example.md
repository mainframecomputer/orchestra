# Composio Adapter

This example demonstrates how to use the Composio adapter with Mainframe Orchestra to access Composio powered intergations.

### Configuration

You will need a Composio API key (get one at [composio.dev](https://composio.dev)) set to COMPOSIO_API_KEY or provde during config

```python
from mainframe_orchestra.adapters.composio_adapter import ComposioAdapter
# Initialize the ComposioAdapter with your API key (or it will use the COMPOSIO_API_KEY env variable)
composio = ComposioAdapter(api_key="YOUR_KEY")
```

### Usage

#### Connecting to Composio apps
Creating a [Composio connection](https://docs.composio.dev/concepts/auth#connection) which connects your user to a [Composio intergation](https://docs.composio.dev/concepts/auth#integration).
The connection_id will be used by Composio to identify the authenticated user.
```python
from composio import App
connection_id, redirect_url = composio.connect(APP.SLACK)
```

#### Getting tools

```python
slack_tools = composio.get_tools_by_app(App.SLACK, connection_id=connection_id)

# if no connection id is provided, defaut user will be used
google_doc_tools = composio.get_tools_by_app(App.GOOGLEDOCS)
```

You can also fetch only a subset of actions from an app
```python
slack_tools = composio.get_tools_by_actions([Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])
```

#### Creating agents
```python

slack = Agent(
    agent_id="slack",
    role="Slack agent",
    goal="Control the Slack app to send and receive messages",
    attributes="You have access to the API provided by Slack. You can help users send and receive messages to a channel and to a single person.",
    llm=OpenaiModels.gpt_4o,
    tools=slack_tools
)

google_docs_agent = Agent(
    agent_id="google_docs",
    role="Google Docs agent",
    goal="Control Google Docs to create and edit documents",
    attributes="You have access to the Google Docs API. You can help users create and edit documents.",
    llm=OpenaiModels.gpt_4o,
    tools=google_doc_tools
)

# Create a conductor agent that can orchestrate the other agents
conductor = Agent(
    agent_id="conductor",
    role="Workflow Conductor",
    goal="Complete user requests by orchestrating specialized agents",
    attributes="You coordinate between different tools to accomplish complex tasks.",
    llm=OpenaiModels.gpt_4o,
    tools=[Conduct.conduct_tool(slack_agent, google_docs_agent)]
)

# Example task that uses the conductor to handle a user request
def document_and_notify_task():
    return Task.create(
        agent=conductor,
        instruction="Create a new Google Doc with meeting notes and then notify the team in Slack that the notes are ready."
    )

# Run the task
response = document_and_notify_task()
print(f"Task completed: {response}")

```
