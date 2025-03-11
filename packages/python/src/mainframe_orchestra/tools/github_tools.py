from typing import Dict, Any, List, Optional
import requests
import base64
import os
from ..utils.braintrust_utils import traced

# Define custom tools
class GitHubTools:
    _owner: Optional[str] = None
    _repo: Optional[str] = None

    @traced(type="tool")
    @staticmethod
    def configure() -> None:
        """
        Configure the GitHub repository details from environment variables.
        Expects GITHUB_OWNER and GITHUB_REPO to be set in environment.

        Raises:
            ValueError: If required environment variables are not set
        """
        owner = os.getenv('GITHUB_OWNER')
        repo = os.getenv('GITHUB_REPO')

        if not owner or not repo:
            raise ValueError("GITHUB_OWNER and GITHUB_REPO environment variables must be set")

        GitHubTools._owner = owner
        GitHubTools._repo = repo

    @traced(type="tool")
    @staticmethod
    def _get_headers(auth_required: bool = False):
        """
        Get headers with or without token.

        Args:
            auth_required (bool): If True, will raise error when token is missing.
                                Use for operations that always need authentication.
        """
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv('GITHUB_TOKEN')
        if auth_required and not token:
            raise ValueError("This operation requires GITHUB_TOKEN environment variable")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


    @traced(type="tool")
    @staticmethod
    def get_user_info(username: str) -> Dict[str, Any]:
        """
        Get public information about a GitHub user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            Dict[str, Any]: User's public information.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/users/{username}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def list_user_repos(username: str) -> List[Dict[str, Any]]:
        """
        List public repositories for the specified user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            List[Dict[str, Any]]: List of repository information.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/users/{username}/repos"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def get_repo_details(owner: str, repo: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific GitHub repository.

        Args:
            owner (str): Repository owner username/organization.
            repo (str): Repository name.

        Returns:
            Dict[str, Any]: Repository details.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def list_repo_contributors(owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        List contributors to a specific GitHub repository.

        Args:
            owner (str): Repository owner username/organization.
            repo (str): Repository name.

        Returns:
            List[Dict[str, Any]]: List of contributor information.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{owner}/{repo}/contributors"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def get_repo_readme(owner: str, repo: str) -> Dict[str, str]:
        """
        Get the README content of a GitHub repository.

        Args:
            owner (str): Repository owner username/organization.
            repo (str): Repository name.

        Returns:
            Dict[str, str]: README content.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{owner}/{repo}/readme"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return {
            "content": base64.b64decode(data["content"]).decode('utf-8'),
            "path": data["path"],
            "name": data["name"]
        }

    @traced(type="tool")
    @staticmethod
    def search_repositories(query: str, sort: str = "stars", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for repositories on GitHub.

        Args:
            query (str): Search keywords and qualifiers.
            sort (str): Sort by: stars, forks, help-wanted-issues, updated.
            max_results (int): Maximum number of results to return.

        Returns:
            Dict[str, Any]: Search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 100)
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "total_count": data["total_count"],
            "incomplete_results": data["incomplete_results"],
            "items": data["items"][:max_results]
        }

    @traced(type="tool")
    @staticmethod
    def search_code(query: str, owner: str = None, repo: str = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for code on GitHub.

        Args:
            query (str): Search keywords and qualifiers.
            owner (str, optional): Limit search to specific owner.
            repo (str, optional): Limit search to specific repo (requires owner).
            max_results (int): Maximum number of results to return.

        Returns:
            Dict[str, Any]: Search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        if repo and not owner:
            raise ValueError("Repository name provided without owner")

        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()

        # Add repo scope if provided
        if owner and repo:
            query = f"{query} repo:{owner}/{repo}"
        elif owner:
            query = f"{query} user:{owner}"

        url = f"{base_url}/search/code"
        params = {
            "q": query,
            "per_page": min(max_results, 100)
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "total_count": data["total_count"],
            "incomplete_results": data["incomplete_results"],
            "items": data["items"][:max_results]
        }

    @traced(type="tool")
    @staticmethod
    def list_pull_requests(state: str = "open") -> List[Dict[str, Any]]:
        """
        List pull requests in a repository.

        Args:
            state (str, optional): State of PRs to return ('open', 'closed', 'all'). Defaults to 'open'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing pull request information.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/pulls"
        params = {"state": state}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        def simplify_pr(pr: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "number": pr["number"],
                "title": pr["title"],
                "state": pr["state"],
                "created_at": pr["created_at"],
                "updated_at": pr["updated_at"],
                "html_url": pr["html_url"],
                "user": {
                    "login": pr["user"]["login"],
                    "id": pr["user"]["id"]
                },
                "head": {
                    "ref": pr["head"]["ref"],
                    "sha": pr["head"]["sha"]
                },
                "base": {
                    "ref": pr["base"]["ref"],
                    "sha": pr["base"]["sha"]
                },
                "mergeable_state": pr.get("mergeable_state"),
                "draft": pr["draft"]
            }

        return [simplify_pr(pr) for pr in response.json()]

    @traced(type="tool")
    @staticmethod
    def get_pull_request(pull_number: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific pull request.

        Args:
            pull_number (int): The pull request number.

        Returns:
            Dict[str, Any]: A dictionary containing detailed pull request information.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/pulls/{pull_number}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def list_pull_request_commits(pull_number: int) -> List[Dict[str, Any]]:
        """
        List commits on a pull request.

        Args:
            pull_number (int): The pull request number.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing commit information.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/pulls/{pull_number}/commits"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        def simplify_commit(commit: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "sha": commit["sha"],
                "message": commit["commit"]["message"],
                "author": {
                    "name": commit["commit"]["author"]["name"],
                    "email": commit["commit"]["author"]["email"],
                    "date": commit["commit"]["author"]["date"]
                },
                "url": commit["html_url"]
            }

        return [simplify_commit(commit) for commit in response.json()]

    @traced(type="tool")
    @staticmethod
    def list_pull_request_files(pull_number: int) -> List[Dict[str, Any]]:
        """
        List files changed in a pull request.

        Args:
            pull_number (int): The pull request number.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing file change information.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/pulls/{pull_number}/files"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        def simplify_file(file: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "filename": file["filename"],
                "status": file["status"],
                "additions": file["additions"],
                "deletions": file["deletions"],
                "changes": file["changes"],
                "blob_url": file["blob_url"],
                "raw_url": file["raw_url"],
                "patch": file.get("patch")
            }

        return [simplify_file(file) for file in response.json()]

    @traced(type="tool")
    @staticmethod
    def get_directory_structure(path: str = "") -> Dict[str, Any]:
        """
        Get the directory structure of a repository.

        Args:
            path (str, optional): The directory path. Defaults to root directory.

        Returns:
            Dict[str, Any]: A nested dictionary representing the directory structure.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        contents = GitHubTools.get_repo_contents(path)
        structure = {}
        for item in contents:
            if item['type'] == 'dir':
                structure[item['name']] = GitHubTools.get_directory_structure(item['path'])
            else:
                structure[item['name']] = item['type']
        return structure

    @traced(type="tool")
    @staticmethod
    def get_repo_contents(path: str = "") -> List[Dict[str, Any]]:
        """
        Get contents of a repository directory or file.

        Args:
            path (str, optional): The directory or file path. Defaults to root directory.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about the contents,
                                 including file SHA if it's a file.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        contents = response.json()
        # If it's a single file, wrap it in a list for consistent handling
        if not isinstance(contents, list):
            contents = [contents]

        return [{
            "name": item["name"],
            "path": item["path"],
            "sha": item["sha"],
            "type": item["type"],
            "size": item.get("size"),
            "download_url": item.get("download_url")
        } for item in contents]

    @traced(type="tool")
    @staticmethod
    def get_file_content(path: str) -> Dict[str, Any]:
        """
        Get the content of a specific file in the repository.

        Args:
            path (str): The file path within the repository.

        Returns:
            Dict[str, Any]: Dictionary containing file content and metadata including SHA

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        return {
            "content": base64.b64decode(data["content"]).decode('utf-8'),
            "sha": data["sha"],
            "size": data["size"],
            "name": data["name"],
            "path": data["path"]
        }

    @traced(type="tool")
    @staticmethod
    def get_issue_comments(issue_number: int) -> List[Dict[str, Any]]:
        """
        Get essential information about an issue and its comments in a repository.

        Args:
            issue_number (int): The number of the issue.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, containing the issue description and all comments.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()

        # Get issue details
        issue_url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/issues/{issue_number}"
        issue_response = requests.get(issue_url, headers=headers)
        issue_response.raise_for_status()
        issue_data = issue_response.json()

        # Get comments
        comments_url = f"{issue_url}/comments"
        comments_response = requests.get(comments_url, headers=headers)
        comments_response.raise_for_status()
        comments_data = comments_response.json()

        def simplify_data(data: Dict[str, Any], is_issue: bool = False) -> Dict[str, Any]:
            return {
                "id": data["id"],
                "user": {
                    "login": data["user"]["login"],
                    "id": data["user"]["id"]
                },
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "body": data["body"],
                "type": "issue" if is_issue else "comment"
            }

        result = [simplify_data(issue_data, is_issue=True)]
        result.extend([simplify_data(comment) for comment in comments_data])

        return result

    @traced(type="tool")
    @staticmethod
    def create_issue_comment(issue_number: int, body: str) -> Dict[str, Any]:
        """
        Create a comment on an issue.

        Args:
            issue_number (int): The issue number to comment on.
            body (str): The comment text.

        Returns:
            Dict[str, Any]: The created comment data.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers(auth_required=True)
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/issues/{issue_number}/comments"

        data = {"body": body}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def list_repo_issues(state: str = "open") -> List[Dict[str, Any]]:
        """
        List issues in the specified repository.

        Args:
            state (str, optional): The state of the issues to return. Can be either 'open', 'closed', or 'all'. Defaults to 'open'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing essential information about an issue.

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails.
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/issues"
        params = {"state": state}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        def simplify_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "number": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "html_url": issue["html_url"],
                "user": {
                    "login": issue["user"]["login"],
                    "id": issue["user"]["id"]
                },
                "comments": issue["comments"],
                "pull_request": "pull_request" in issue
            }

        return [simplify_issue(issue) for issue in response.json()]

    @traced(type="tool")
    @staticmethod
    def check_github_diff(base: str, head: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check differences between two Git references (commits, branches, or tags).

        Args:
            base (str): Base reference (commit SHA, branch, or tag)
            head (str): Head reference to compare against base
            file_path (Optional[str]): Optional specific file path to get diff for

        Returns:
            Dict[str, Any]: A dictionary containing:
                - files_changed: List of changed files
                - total_changes: Stats about insertions/deletions
                - diff: The actual diff content
                - commits: List of commits between references

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        headers = GitHubTools._get_headers(auth_required=True)

        # Construct URL for the comparison
        url = f"https://api.github.com/repos/{GitHubTools._owner}/{GitHubTools._repo}/compare/{base}...{head}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"GitHub API error ({response.status_code}): {response.text}")

        data = response.json()

        # Process the response
        result = {
            "files_changed": [],
            "total_changes": {
                "additions": 0,
                "deletions": 0,
                "changes": 0
            },
            "diff": "",
            "commits": []
        }

        # Get diff content if specific file is requested
        if file_path:
            diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
            diff_response = requests.get(url, headers=diff_headers)
            full_diff = diff_response.text
            # Extract specific file diff (basic implementation)
            for diff_section in full_diff.split("diff --git"):
                if file_path in diff_section:
                    result["diff"] = diff_section
                    break
        else:
            # Get full diff
            diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
            diff_response = requests.get(url, headers=diff_headers)
            result["diff"] = diff_response.text

        # Process files and stats
        for file in data.get("files", []):
            result["files_changed"].append(file["filename"])
            result["total_changes"]["additions"] += file.get("additions", 0)
            result["total_changes"]["deletions"] += file.get("deletions", 0)
            result["total_changes"]["changes"] += file.get("changes", 0)

        # Process commits
        for commit in data.get("commits", []):
            result["commits"].append({
                "sha": commit["sha"],
                "message": commit["commit"]["message"],
                "author": commit["commit"]["author"]["name"]
            })

        return result

    @traced(type="tool")
    @staticmethod
    def update_file(path: str, message: str, content: str, branch: str) -> Dict[str, Any]:
        """
        Create or update a file in the repository.
        Will automatically fetch the file's current SHA if it exists.

        Args:
            path (str): Path to the file
            message (str): Commit message
            content (str): New file content
            branch (str): Branch name

        Returns:
            Dict[str, Any]: Response data including commit details

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        headers = GitHubTools._get_headers(auth_required=True)
        base_url = "https://api.github.com"

        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/contents/{path}"

        # Try to get existing file's SHA
        try:
            current_file = GitHubTools.get_file_content(path)
            sha = current_file["sha"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:  # If error is not "file doesn't exist"
                raise
            sha = None

        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }

        if sha:
            data["sha"] = sha

        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def get_default_branch() -> str:
        """
        Get the default branch (usually main or master) of the repository

        Returns:
            str: Name of the default branch

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers()
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()["default_branch"]

    @traced(type="tool")
    @staticmethod
    def create_branch(branch_name: str) -> Dict[str, Any]:
        """
        Create a new branch from the default branch.

        Args:
            branch_name (str): Name for the new branch

        Returns:
            Dict[str, Any]: Response data including ref details

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        # Get the default branch's SHA
        default_branch = GitHubTools.get_default_branch()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers(auth_required=True)

        # Get the SHA of the default branch's HEAD
        ref_url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/git/ref/heads/{default_branch}"
        ref_response = requests.get(ref_url, headers=headers)
        ref_response.raise_for_status()
        base_sha = ref_response.json()["object"]["sha"]

        # Create new branch
        create_url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/git/refs"
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }

        response = requests.post(create_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def commit_file(path: str, content: str, message: str, branch: str) -> Dict[str, Any]:
        """
        Commit a single file change to a branch.

        Args:
            path (str): Path to the file
            content (str): New file content
            message (str): Commit message
            branch (str): Branch name

        Returns:
            Dict[str, Any]: Response data including commit details

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers(auth_required=True)

        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/contents/{path}"

        # Get current file info if it exists
        try:
            current_file = GitHubTools.get_file_content(path)
            sha = current_file["sha"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:
                raise
            sha = None

        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }

        if sha:
            data["sha"] = sha

        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    @traced(type="tool")
    @staticmethod
    def create_pull_request(title: str, body: str, head: str, base: str = None) -> Dict[str, Any]:
        """
        Create a new pull request.

        Args:
            title (str): PR title
            body (str): PR description
            head (str): Name of the branch with changes
            base (str, optional): Name of the branch to merge into. Defaults to repo's default branch.

        Returns:
            Dict[str, Any]: Pull request data

        Raises:
            ValueError: If owner and repo are not configured
            requests.exceptions.HTTPError: If the API request fails
        """
        GitHubTools.configure()
        base_url = "https://api.github.com"
        headers = GitHubTools._get_headers(auth_required=True)
        url = f"{base_url}/repos/{GitHubTools._owner}/{GitHubTools._repo}/pulls"

        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

