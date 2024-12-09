# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import requests
from typing import Dict, Any, List
import base64

class GitHubTools:

    @staticmethod
    def get_user_info(username: str) -> Dict[str, Any]:
        """
        Get public information about a GitHub user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            Dict[str, Any]: A dictionary containing the user's public information.
                            Keys may include 'login', 'id', 'name', 'company', 'blog',
                            'location', 'email', 'hireable', 'bio', 'public_repos',
                            'public_gists', 'followers', 'following', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/users/{username}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def list_user_repos(username: str) -> List[Dict[str, Any]]:
        """
        List public repositories for the specified user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information
                                  about a public repository. Keys may include 'id',
                                  'node_id', 'name', 'full_name', 'private', 'owner',
                                  'html_url', 'description', 'fork', 'url', 'created_at',
                                  'updated_at', 'pushed_at', 'homepage', 'size',
                                  'stargazers_count', 'watchers_count', 'language',
                                  'forks_count', 'open_issues_count', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/users/{username}/repos"
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        return response.json()

    @staticmethod
    def list_repo_issues(owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """
        List issues in the specified public repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            state (str, optional): The state of the issues to return. Can be either 'open', 'closed', or 'all'. Defaults to 'open'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing essential information about an issue.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/issues"
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
    
    @staticmethod
    def get_issue_comments(owner: str, repo: str, issue_number: int) -> List[Dict[str, Any]]:
        """
        Get essential information about an issue and its comments in a repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            issue_number (int): The number of the issue.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, containing the issue description and all comments.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        
        # Get issue details
        issue_url = f"{base_url}/repos/{owner}/{repo}/issues/{issue_number}"
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

    @staticmethod
    def get_repo_details(owner: str, repo: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            Dict[str, Any]: A dictionary containing detailed information about the repository.
                            Keys may include 'id', 'node_id', 'name', 'full_name', 'private',
                            'owner', 'html_url', 'description', 'fork', 'url', 'created_at',
                            'updated_at', 'pushed_at', 'homepage', 'size', 'stargazers_count',
                            'watchers_count', 'language', 'forks_count', 'open_issues_count',
                            'master_branch', 'default_branch', 'topics', 'has_issues', 'has_projects',
                            'has_wiki', 'has_pages', 'has_downloads', 'archived', 'disabled', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def list_repo_contributors(owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        List contributors to a specific GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information about a contributor.
                                  Keys may include 'login', 'id', 'node_id', 'avatar_url', 'gravatar_id',
                                  'url', 'html_url', 'followers_url', 'following_url', 'gists_url',
                                  'starred_url', 'subscriptions_url', 'organizations_url', 'repos_url',
                                  'events_url', 'received_events_url', 'type', 'site_admin',
                                  'contributions', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/contributors"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_repo_readme(owner: str, repo: str) -> Dict[str, str]:
        """
        Get the README content of a GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            Dict[str, str]: A dictionary containing the README content.
                            The key is 'content' and the value is the raw text of the README file.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Note:
            This method retrieves the raw content of the README file, regardless of its format
            (e.g., .md, .rst, .txt). The content is not rendered or processed in any way.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/readme"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {"content": response.text}
    
    @staticmethod
    def search_repositories(query: str, sort: str = "stars", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for repositories on GitHub with a maximum number of results.
        
        Args:
            query (str): Search keywords and qualifiers.
            sort (str): Can be one of: stars, forks, help-wanted-issues, updated. Default: stars
            max_results (int): Maximum number of results to return. Default: 10

        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 100)  # GitHub API allows max 100 items per page
        }
        
        results = []
        while len(results) < max_results:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results.extend(data['items'][:max_results - len(results)])
            
            if 'next' not in response.links:
                break
            
            url = response.links['next']['url']
            params = {}  # Clear params as they're included in the next URL

        def simplify_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": {
                    "login": repo["owner"]["login"],
                    "id": repo["owner"]["id"]
                },
                "html_url": repo["html_url"],
                "description": repo["description"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "stargazers_count": repo["stargazers_count"],
                "forks_count": repo["forks_count"],
                "language": repo["language"],
                "topics": repo["topics"],
                "license": repo["license"]["name"] if repo["license"] else None,
                "open_issues_count": repo["open_issues_count"]
            }

        simplified_results = [simplify_repo(repo) for repo in results]

        return {
            "total_count": data['total_count'],
            "incomplete_results": data['incomplete_results'],
            "items": simplified_results[:max_results]
        }


    @staticmethod
    def get_repo_contents(owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """
        Get contents of a repository directory or file.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str, optional): The directory or file path. Defaults to root directory.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about the contents.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_file_content(owner: str, repo: str, path: str) -> str:
        """
        Get the content of a specific file in the repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str): The file path within the repository.

        Returns:
            str: The content of the file.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()["content"]
        return base64.b64decode(content).decode('utf-8')

    @staticmethod
    def get_directory_structure(owner: str, repo: str, path: str = "") -> Dict[str, Any]:
        """
        Get the directory structure of a repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str, optional): The directory path. Defaults to root directory.

        Returns:
            Dict[str, Any]: A nested dictionary representing the directory structure.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        contents = GitHubTools.get_repo_contents(owner, repo, path)
        structure = {}
        for item in contents:
            if item['type'] == 'dir':
                structure[item['name']] = GitHubTools.get_directory_structure(owner, repo, item['path'])
            else:
                structure[item['name']] = item['type']
        return structure

    @staticmethod
    def search_code(query: str, owner: str, repo: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for code within a specific repository.

        Args:
            query (str): The search query.
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            Dict[str, Any]: A dictionary containing search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/search/code"
        params = {
            "q": f"{query} repo:{owner}/{repo}",
            "per_page": min(max_results, 100)
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        def simplify_code_result(item: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "name": item["name"],
                "path": item["path"],
                "sha": item["sha"],
                "url": item["html_url"],
                "repository": {
                    "name": item["repository"]["name"],
                    "full_name": item["repository"]["full_name"],
                    "owner": item["repository"]["owner"]["login"]
                }
            }

        simplified_results = [simplify_code_result(item) for item in data['items'][:max_results]]

        return {
            "total_count": data['total_count'],
            "incomplete_results": data['incomplete_results'],
            "items": simplified_results
        }

