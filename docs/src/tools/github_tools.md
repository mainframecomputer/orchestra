# GitHub Tools

The GitHubTools class provides a comprehensive set of methods for interacting with the GitHub API. It allows you to retrieve user information, repository details, search for repositories and code, manage issues and pull requests, and perform various file operations.

### Configuration

Before using repository-specific operations, configure the target repository:

```python
GitHubTools.configure("owner", "repo")
```

### Authentication

Many operations require authentication. Set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN=your_github_token
```

### Class Methods

##### get_user_info(username: str)

Retrieves public information about a GitHub user, including their login, id, name, company, blog, location, email, bio, and various repository and follower counts.

```python
user_info = GitHubTools.get_user_info("octocat")
```

##### list_user_repos(username: str)

Lists public repositories for a specified user, providing detailed information about each repository, including its name, description, creation date, star count, and more.

```python
user_repos = GitHubTools.list_user_repos("octocat")
```

##### list_repo_issues(owner: str, repo: str, state: str = "open")

Lists issues in a specified public repository. You can filter issues by state (open, closed, or all). The method returns simplified issue information, including the issue number, title, state, creation date, and associated user.

```python
repo_issues = GitHubTools.list_repo_issues("octocat", "Hello-World", state="all")
```

##### get_issue_comments(owner: str, repo: str, issue_number: int)

Retrieves the issue description and all comments for a specific issue in a repository. This method provides a comprehensive view of the discussion around an issue.

```python
issue_comments = GitHubTools.get_issue_comments("octocat", "Hello-World", 1)
```

##### get_repo_details(owner: str, repo: str)

Fetches detailed information about a specific GitHub repository, including its name, description, creation date, star count, fork count, primary language, and various repository settings.

```python
repo_details = GitHubTools.get_repo_details("octocat", "Hello-World")
```

##### list_repo_contributors(owner: str, repo: str)

Lists contributors to a specific GitHub repository, providing information about each contributor, including their username, id, avatar URL, and contribution count.

```python
repo_contributors = GitHubTools.list_repo_contributors("octocat", "Hello-World")
```

##### get_repo_readme(owner: str, repo: str)

Retrieves the raw content of a repository's README file, regardless of its format (e.g., .md, .rst, .txt).

```python
readme_content = GitHubTools.get_repo_readme("octocat", "Hello-World")
```

##### search_repositories(query: str, sort: str = "stars", max_results: int = 10)

Searches for repositories on GitHub based on a query, with options to sort results and limit the number of results returned. The method provides simplified information about each matching repository.

```python
search_results = GitHubTools.search_repositories("machine learning", sort="stars", max_results=5)
```

##### get_repo_contents(owner: str, repo: str, path: str = "")

Retrieves the contents of a repository directory or file. This method is useful for exploring the structure of a repository.

```python
repo_contents = GitHubTools.get_repo_contents("octocat", "Hello-World", "src")
```

##### get_file_content(owner: str, repo: str, path: str)

Fetches the content of a specific file in a repository. This method is useful for accessing the raw content of individual files.

```python
file_content = GitHubTools.get_file_content("octocat", "Hello-World", "README.md")
```

##### get_directory_structure(owner: str, repo: str, path: str = "")

Generates a nested dictionary representing the directory structure of a repository. This method is helpful for understanding the layout of a repository.

```python
directory_structure = GitHubTools.get_directory_structure("octocat", "Hello-World")
```

##### search_code(query: str, owner: str, repo: str, max_results: int = 10)

Searches for code within a specific repository based on a query. This method is useful for finding specific code snippets or files within a repository.

```python
code_search_results = GitHubTools.search_code("def main", "octocat", "Hello-World", max_results=5)
```

##### list_pull_requests(state: str = "open")

Lists pull requests in the configured repository. Returns simplified PR information including number, title, state, and associated metadata.

```python
pull_requests = GitHubTools.list_pull_requests(state="open")
```

##### get_pull_request(pull_number: int)

Gets detailed information about a specific pull request.

```python
pr_details = GitHubTools.get_pull_request(42)
```

##### list_pull_request_commits(pull_number: int)

Lists commits in a specific pull request with simplified commit information.

```python
pr_commits = GitHubTools.list_pull_request_commits(42)
```

##### list_pull_request_files(pull_number: int)

Lists files changed in a pull request with details about additions, deletions, and patches.

```python
pr_files = GitHubTools.list_pull_request_files(42)
```

##### create_issue_comment(issue_number: int, body: str)

Creates a new comment on an issue or pull request.

```python
comment = GitHubTools.create_issue_comment(123, "Great work on this!")
```

##### check_github_diff(base: str, head: str, file_path: Optional[str] = None)

Compares two Git references and returns difference information.

```python
diff = GitHubTools.check_github_diff("main", "feature-branch", "path/to/file.py")
```

##### update_file(path: str, message: str, content: str, branch: str)

Creates or updates a file in the repository.

```python
result = GitHubTools.update_file(
    "docs/README.md",
    "Update documentation",
    "# New content",
    "main"
)
```

##### get_default_branch()

Gets the default branch name (usually main or master) of the configured repository.

```python
default_branch = GitHubTools.get_default_branch()
```

##### create_branch(branch_name: str)

Creates a new branch from the default branch.

```python
new_branch = GitHubTools.create_branch("feature/new-feature")
```

##### commit_file(path: str, content: str, message: str, branch: str)

Commits a single file change to a branch.

```python
result = GitHubTools.commit_file(
    "src/main.py",
    "print('Hello, World!')",
    "Add hello world example",
    "feature-branch"
)
```

##### create_pull_request(title: str, body: str, head: str, base: str = None)

Creates a new pull request. If base is not specified, uses the repository's default branch.

```python
pr = GitHubTools.create_pull_request(
    "Add new feature",
    "This PR implements...",
    "feature-branch"
)
```

### Usage Notes

1. **Repository Configuration**: Many methods require prior configuration using `GitHubTools.configure()`. Ensure this is called before using repository-specific operations.

2. **Authentication**: Some operations require authentication via a GitHub token. Set the `GITHUB_TOKEN` environment variable before using these methods.

3. **Error Handling**: Methods raise appropriate exceptions:
   - `ValueError` when repository configuration is missing
   - `requests.exceptions.HTTPError` for API failures
   - Custom exceptions for specific scenarios

4. **Rate Limiting**: Be mindful of GitHub API rate limits. Authenticated requests have higher limits than unauthenticated ones.

5. **Branch Operations**: When working with branches and files:
   - Always verify the target branch exists
   - Consider using `get_default_branch()` when unsure about the base branch
   - Handle file conflicts appropriately

6. **Pull Requests**: When creating PRs:
   - Ensure branches exist and have commits
   - Provide clear titles and descriptions
   - Consider using the simplified PR information methods for better performance

The GitHubTools class provides a pythonic interface to common GitHub operations while handling authentication, error checking, and response processing automatically.

