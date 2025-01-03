from typing import Dict, Any
from mainframe_orchestra import Agent, Task, OpenaiModels, GitHubTools, set_verbosity
from fastapi import FastAPI, Request
import uvicorn
set_verbosity(0)

# This example creates an orchestra of agents that automatically analyzes and responds to repository events through webhooks. It processes three types of events:

# 1. Pull Requests: 
#    - Analyzes new/updated PRs
#    - Reviews code changes and diffs
#    - Identifies potential issues or mistakes

# 2. Issues:
#    - Reviews new/updated issues
#    - Examines mentioned code files
#    - Analyzes error messages and logs
#    - Suggests potential solutions

# 3. Issue Comments:
#    - Tracks ongoing discussions
#    - Updates analysis based on new information
#    - Reviews any new files or errors mentioned
#    - Provides focused responses to new context

# Required environment variables:
# - GITHUB_TOKEN: GitHub API authentication token, optional (increases rate limit and permits private repo access, make sure you set necessary permissions on repo in github settings)
# - GITHUB_WEBHOOK_SECRET: Secret for webhook validation (create your own random string and set it in GitHub repo webhook settings)
# - GITHUB_WEBHOOK_PORT: Port number for webhook server (default is 8080), use ngrok to test locally
# - GITHUB_WEBHOOK_ENABLED: Set to true to enable webhooks
# - GITHUB_OWNER: Owner of the repository
# - GITHUB_REPO: Name of the repository

# step 1: create webhook in github repo settings
# step 2: run ngrok to get the url
# step 3: paste ngrok url into github webhook settings
# step 4: create a random string for the webhook secret and set in environment variables and github webhook settings

github_agent = Agent(
    agent_id="github_agent",
    role="GitHub Agent",
    goal="Analyze GitHub pull requests and issues, and investigate them",
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
                
            if event_type == 'issue_comment':
                print(f"Processing issue comment webhook: {payload.get('action')}")
                response = await handle_issue_comment_webhook(payload)
                return {"status": "processed", "response": str(response)}
                
            elif event_type == 'issues':
                print(f"Processing issue webhook: {payload.get('action')}")
                response = await handle_issue_webhook(payload)
                return {"status": "processed", "response": str(response)}
            
            elif event_type == 'pull_request':
                print(f"Processing PR webhook: {payload.get('action')}")
                response = await handle_pr_webhook(payload)
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

async def handle_pr_webhook(payload: Dict[str, Any]) -> str:
    """Handle Pull Request webhook events with appropriate context and actions"""
    pr_action = payload.get("action")
    pr = payload.get("pull_request", {})
    owner = payload.get("repository", {}).get("owner", {}).get("login")
    repo = payload.get("repository", {}).get("name")
    pr_number = pr.get("number")

    pr_context = {
        "action": pr_action,
        "pull_request": pr,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number
    }

    response = await Task.create(
        agent=github_agent,
        instruction=f""""Review PR #{pr_number} {pr_action} in {owner}/{repo}. 
~~~
PR information:
{pr_context}
~~~  
Use the tools to get the full context of the PR and the codebase. Investigate the PR and provide a detailed analysis and check the diff and pull request for any silly mistakes or errors. Provide a detailed analysis of the PR and the codebase and any potential issues.
"""
    )
    print(response)

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
    
    instruction = f"""You are reviewing issue #{issue_number} in {owner}/{repo}. 
~~~
Issue information:
{issue_context}
~~~    
First, fetch the full conversation history. Then, if any code files are mentioned, examine them. Examine the technical details, including any error messages, logs, or code snippets provided. If specific files are mentioned, review their contents. Finally, provide your assessment of the root cause and suggest potential solutions.
"""
    
    response = await Task.create(
        agent=github_agent,
        instruction=instruction
    )
    print(response)

async def handle_issue_comment_webhook(payload: Dict[str, Any]) -> str:
    """Handle Issue Comment webhook events with appropriate context and actions"""
    action = payload.get("action")
    comment = payload.get("comment", {})
    issue = payload.get("issue", {})
    owner = payload.get("repository", {}).get("owner", {}).get("login")
    repo = payload.get("repository", {}).get("name")
    issue_number = issue.get("number")
    
    issue_comment_context = {
        "action": action,
        "comment": {
            "body": comment.get("body"),
            "user": comment.get("user", {}).get("login"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "html_url": comment.get("html_url")
        },
        "issue": {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "state": issue.get("state"),
            "html_url": issue.get("html_url")
        },
        "repository": f"{owner}/{repo}"
    }
    
    instruction = f"""You are a technical analyst following issue #{issue_number} in {owner}/{repo}. 
~~~
Issue information:
{issue_comment_context}
~~~   
1. Check if the comment:
   - Provides new error information or logs
   - Mentions new files to investigate
   - Reports success/failure of previous suggestions
   - Asks for clarification
   - Provides additional context

2. If needed:
   - Use get_issue_comments to review the full conversation
   - Use get_file_content if new files are mentioned
   - Compare new error reports with previous ones

3. Focus on what's new or different in this comment compared to previous discussion.

Provide a focused response based on the new information and whether it changes the understanding of the issue."""
    
    response = await Task.create(
        agent=github_agent,
        instruction=instruction
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
