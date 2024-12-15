# GitHub Tools

The GitHubTools class provides a comprehensive set of methods for interacting with the GitHub API. It allows you to retrieve user information, repository details, search for repositories and code, manage issues, and perform various file operations.

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

### Usage Notes

When using the GitHubTools class, ensure that you have a stable internet connection to make API requests. The methods raise appropriate exceptions (requests.exceptions.HTTPError) if the API requests fail, so make sure to handle them accordingly in your code.

The GitHubTools class provides a set of static methods, which means you can directly call them using the class name without creating an instance of the class.

Remember to adhere to the GitHub API rate limits and usage guidelines when making requests. Excessive or abusive requests may result in temporary or permanent restrictions on your API access.

Some methods, like search_repositories() and search_code(), allow you to specify a maximum number of results. This is useful for managing the amount of data returned and controlling API usage.

When working with repository contents and file operations, be mindful of the repository size and structure. For large repositories, it may be more efficient to use targeted methods like get_file_content() rather than fetching the entire directory structure.

The GitHubTools class simplifies many common GitHub API operations, but for more advanced use cases, you may need to refer to the official GitHub API documentation and potentially extend the class with additional methods.

