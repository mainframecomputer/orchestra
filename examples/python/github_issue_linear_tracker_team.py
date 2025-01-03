from typing import Dict, Any
from mainframe_orchestra import Agent, Task, Conduct, OpenaiModels, GitHubTools, LinearTools, set_verbosity
from fastapi import FastAPI, Request
import uvicorn
set_verbosity(0)

# This example demonstrates an automated integration between GitHub Issues and Linear tickets using Orchestra agents.
# NOTE: This team is capable of creating new issues and updating issue statuses in Linear. Use in a test team, or remove the update_issue_status tool and create_issue tool if you don't want to edit issues in Linear.

# When a GitHub issue is created or updated, the system:
# 1. Receives the webhook from GitHub
# 2. Uses a GitHub agent to analyze the issue, its comments, and related code
# 3. Uses a Linear agent to create/update corresponding tickets in Linear
# 4. Coordinates these actions through a coordinator agent

# Required environment variables:
# - GITHUB_TOKEN: GitHub API authentication token, optional (increases rate limit and permits private repo access, make sure you set necessary permissions on repo in github settings)
# - GITHUB_WEBHOOK_SECRET: Secret for webhook validation (create a random string and set it in GitHub repo webhook settings)
# - GITHUB_WEBHOOK_PORT: Port number for webhook server (default is 8080), use ngrok to test locally
# - GITHUB_WEBHOOK_ENABLED: Set to true to enable webhooks
# - GITHUB_OWNER: Owner of the repository
# - GITHUB_REPO: Name of the repository
# - LINEAR_API_KEY: API key for Linear
# - LINEAR_TEAM_ID: Team ID for Linear

# To run this example, you need to:
# step 1: create webhook in github repo settings
# step 2: run ngrok to get the url
# step 3: paste ngrok url into github webhook settings
# step 4: create a random string for the webhook secret and set in environment variables and github webhook settings

# Initialize the linear toolkit 
linear_tools_instance = LinearTools()

linear_agent = Agent(
    agent_id="linear_agent",
    role="Linear Agent",
    goal="Use your linear tools to assist with the given task",
    tools=[
        linear_tools_instance.get_team_issues, 
        linear_tools_instance.get_workflow_states,
        linear_tools_instance.search_issues,
        linear_tools_instance.update_issue_status,
        linear_tools_instance.create_issue
    ],
    llm=OpenaiModels.gpt_4o
)

github_agent = Agent(
    agent_id="github_agent",
    role="GitHub Agent",
    goal="Analyze GitHub pull requests and issues, and investigate them",
    attributes="You know you should investigate the codebase and relevant files where necessary based on the issue.",
    tools=[
        GitHubTools.get_directory_structure, 
        GitHubTools.get_file_content,
        GitHubTools.get_issue_comments,
        GitHubTools.list_repo_issues,
        GitHubTools.get_pull_request, 
        GitHubTools.list_pull_request_commits, 
        GitHubTools.list_pull_request_files,
        GitHubTools.check_github_diff,
    ],
    llm=OpenaiModels.gpt_4o
)

coordinator = Agent(
    agent_id="coordinator",
    role="Coordinator",
    goal="To chat with and help the human user by coordinating your team of agents to carry out tasks",
    tools=[Conduct.conduct_tool(github_agent, linear_agent)],
    llm=OpenaiModels.gpt_4o
)

class GithubConfig:
    """Configuration class for GitHub integration"""
    def __init__(self, 
                 owner: str = None,
                 repo: str = None,
                 webhook_secret: str = None,
                 webhook_port: int = 8080,
                 webhook_enabled: bool = False):
        self.owner = owner
        self.repo = repo
        self.webhook_secret = webhook_secret
        self.webhook_port = webhook_port
        self.webhook_enabled = webhook_enabled

    @classmethod
    def from_env(cls):
        """Alternative constructor to load config from environment variables"""
        import os
        return cls(
            owner=os.getenv('GITHUB_OWNER'),
            repo=os.getenv('GITHUB_REPO'),
            webhook_secret=os.getenv('GITHUB_WEBHOOK_SECRET'),
            webhook_port=int(os.getenv('GITHUB_WEBHOOK_PORT', '8080')),
            webhook_enabled=os.getenv('GITHUB_WEBHOOK_ENABLED', 'false').lower() == 'true'
        )

class GithubWebhookHandler:
    def __init__(self, secret_token: str = None):
        self.secret_token = secret_token
        self.app = FastAPI(title="GitHub Webhook Handler")
        self.setup_webhook_endpoint()
        
    def setup_webhook_endpoint(self):
        @self.app.post("/webhook")
        async def webhook(request: Request):
            if self.secret_token:
                # Verify signature here if needed
                pass
                
            event_type = request.headers.get('X-GitHub-Event')
            payload = await request.json()
            
            print(f"Received webhook event: {event_type}")  # Debug log
                
            if event_type == 'issues':
                print(f"Processing issue webhook: {payload.get('action')}")
                response = await handle_issue_webhook(payload)
                return {"status": "processed", "response": str(response)}
                
            print(f"Ignoring event type: {event_type}")
            return {"status": "ignored", "event": event_type}

    def start(self, port: int = 8080):
        """Start the webhook server in a separate thread"""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )

async def handle_issue_webhook(payload: Dict[str, Any]) -> str:
    """Handle Issue webhook events with appropriate context and actions"""
    issue_action = payload.get("action")
    issue = payload.get("issue", {})
    owner = payload.get("repository", {}).get("owner", {}).get("login")
    repo = payload.get("repository", {}).get("name")
    issue_number = issue.get("number")
    
    issue_context = {
        "action": issue_action,
        "issue": {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body"),
            "state": issue.get("state"),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "html_url": issue.get("html_url"),
            "user": issue.get("user", {}).get("login")
        },
        "repository": f"{owner}/{repo}"
    }
    
    response = await Task.create(
        agent=coordinator,
        instruction=f"""GitHub issue #{issue_number} in {owner}/{repo} needs to be investigated and tracked. 

Coordinate with the GitHub agent to thoroughly investigate this issue, including its comments, related code, relevant files, and any connected pull requests. The investigation should provide a complete understanding of the issue and the relevant code that may be causing the issue. Then have the Linear agent check for any existing issues for the ticket, and if none exist, create a new issue to track this. The Linear issue should use the original GitHub issue title prefixed with "[GH #{issue_number}]" and include the detailed analysis in the description.

~~~
Issue Context:
{issue_context}
~~~

Orchestrate your team to ensure both the investigation and Linear ticket creation are completed properly.
"""
    )
    print(response)

def main(config: GithubConfig):
    """
    Main function that sets up the GitHub agent and optionally webhooks
    
    Example usage:
    ```python
    # Method 1: Direct configuration
    config = GithubConfig(
        owner="your-org",
        repo="your-repo",
        webhook_enabled=True
    )
    
    # Method 2: Environment variables
    config = GithubConfig.from_env()
    
    # Method 3: Config file
    config = GithubConfig.from_config_file('config.yaml')
    
    main(config)
    ```
    """
    if config.webhook_enabled:
        webhook_handler = GithubWebhookHandler(
            secret_token=config.webhook_secret
        )
        webhook_handler.start(port=config.webhook_port)
        print(f"Webhook server started on port {config.webhook_port}")

if __name__ == "__main__":
    config = GithubConfig.from_env()
    main(config)
