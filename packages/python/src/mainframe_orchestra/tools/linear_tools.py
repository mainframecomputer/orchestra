# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import requests
from typing import Dict, List, Optional, Union
from ..utils.braintrust_utils import traced

class LinearTools:
    def __init__(self):
        """
        Initialize Linear API client with authentication.
        Requires LINEAR_API_KEY and LINEAR_TEAM_ID environment variables to be set.
        """
        self.api_key = os.getenv('LINEAR_API_KEY')
        self.team_id = os.getenv('LINEAR_TEAM_ID')
        if not self.api_key:
            raise ValueError("LINEAR_API_KEY environment variable is required")
        if not self.team_id:
            raise ValueError("LINEAR_TEAM_ID environment variable is required")

        self.url = 'https://api.linear.app/graphql'
        self.headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }

    def _execute_query(self, query: str, variables: Dict = None) -> Dict:
        """
        Execute a GraphQL query using requests
        """
        response = requests.post(
            self.url,
            headers=self.headers,
            json={'query': query, 'variables': variables}
        )
        response.raise_for_status()
        return response.json()

    @classmethod
    @traced(type="tool")
    def get_team_issues(cls, team_id: str = None, status: Optional[str] = None) -> Union[List[Dict], str]:
        """
        Get all issues for a team.
        If team_id is not provided, uses LINEAR_TEAM_ID from env.

        Args:
            team_id (str, optional): Team ID or key. If not provided, uses LINEAR_TEAM_ID from env
            status (str, optional): Status to filter issues by. If not provided, returns all issues

        Returns:
            Union[List[Dict], str]: List of issues if successful, error message if failed
        """
        try:
            client = cls()
            query = """
                query TeamIssues($teamId: String!, $status: String) {
                    team(id: $teamId) {
                        issues(filter: {
                            state: { name: { eq: $status } }
                        }) {
                            nodes {
                                id
                                title
                                state {
                                    name
                                }
                                priority
                                assignee {
                                    name
                                }
                                url
                            }
                        }
                    }
                }
            """

            variables = {"teamId": team_id or client.team_id}
            if status:
                variables["status"] = status

            result = client._execute_query(query, variables)
            return result["data"]["team"]["issues"]["nodes"]
        except Exception as e:
            return f"Error fetching team issues: {str(e)}"

    @classmethod
    @traced(type="tool")
    def update_issue_status(cls, issue_id: str, status_id: str) -> Union[Dict, str]:
        """
        Update the status of an issue.

        Args:
            issue_id (str): ID of the issue to update
            status_id (str): ID of the new status

        Returns:
            Union[Dict, str]: Updated issue data if successful, error message if failed
        """
        try:
            client = cls()  # Create instance within the method
            mutation = """
                mutation UpdateIssueStatus($issueId: String!, $statusId: String!) {
                    issueUpdate(
                        id: $issueId,
                        input: { stateId: $statusId }
                    ) {
                        success
                        issue {
                            id
                            title
                            state {
                                name
                            }
                        }
                    }
                }
            """

            result = client._execute_query(mutation, {
                "issueId": issue_id,
                "statusId": status_id
            })
            return result["data"]["issueUpdate"]
        except Exception as e:
            return f"Error updating issue status: {str(e)}"

    @classmethod
    @traced(type="tool")
    def search_issues(cls, search_query: str) -> Union[List[Dict], str]:
        """
        Search for issues using a text query.

        Args:
            search_query (str): Search query string

        Returns:
            Union[List[Dict], str]: List of matching issues if successful, error message if failed
        """
        try:
            client = cls()  # Create instance within the method
            search_gql = """
                query SearchIssues($query: String!) {
                    issueSearch(query: $query) {
                        nodes {
                            id
                            title
                            description
                            state {
                                name
                            }
                            priority
                            url
                        }
                    }
                }
            """

            result = client._execute_query(search_gql, {
                "query": search_query
            })
            return result["data"]["issueSearch"]["nodes"]
        except Exception as e:
            return f"Error searching issues: {str(e)}"

    @classmethod
    @traced(type="tool")
    def get_team_by_name(cls, team_name: str) -> Union[Dict, str]:
        """
        Get team ID by team name.

        Args:
            team_name (str): Name of the team

        Returns:
            Union[Dict, str]: Team data if successful, error message if failed
        """
        try:
            client = cls()  # Create instance within the method
            query = """
                query GetTeams {
                    teams {
                        nodes {
                            id
                            name
                            key
                        }
                    }
                }
            """

            result = client._execute_query(query)
            teams = result["data"]["teams"]["nodes"]

            print("Available teams:")
            for team in teams:
                print(f"- {team['name']} (key: {team['key']})")

            matching_team = next((team for team in teams if team['key'] == team_name), None)
            if not matching_team:
                return f"No team found with key: {team_name}"

            return matching_team
        except Exception as e:
            return f"Error fetching team: {str(e)}"

    @classmethod
    @traced(type="tool")
    def get_workflow_states(cls, team_id: str = None) -> Union[List[Dict], str]:
        """
        Get all workflow states for a team.
        If team_id is not provided, uses LINEAR_TEAM_ID from env.
        """
        try:
            client = cls()  # Create instance within the method
            query = """
                query WorkflowStates($teamId: String!) {
                    team(id: $teamId) {
                        states {
                            nodes {
                                id
                                name
                                type
                            }
                        }
                    }
                }
            """

            result = client._execute_query(query, {"teamId": team_id or client.team_id})
            return result["data"]["team"]["states"]["nodes"]
        except Exception as e:
            return f"Error fetching workflow states: {str(e)}"

    @traced(type="tool")
    @classmethod
    def create_issue(cls, title: str, description: str, team_id: str = None, priority: Optional[int] = None, state_id: Optional[str] = None) -> Union[Dict, str]:
        """
        Create a new issue in Linear.
        If team_id is not provided, uses TEAM_ID from env.

        Args:
            title (str): Title of the issue
            description (str): Detailed description of the issue
            team_id (str, optional): Team key or ID. If not provided, uses LINEAR_TEAM_ID from env
            priority (int, optional): Priority level (0-4, where 0 is no priority, 4 is urgent)
            state_id (str, optional): ID of the initial state (if not provided, will use team's default)

        Returns:
            Union[Dict, str]: Issue data if successful, error message if failed
        """
        try:
            client = cls()

            # Use the team_id directly if it looks like a UUID
            team_uuid = team_id or client.team_id
            if not team_uuid.startswith('-') and len(team_uuid.split('-')) == 5:
                # Looks like a UUID, use it directly
                pass
            else:
                # Treat as team key and look up the UUID
                team_info = cls.get_team_by_name(team_uuid)
                if isinstance(team_info, str):  # Error message
                    return f"Error getting team info: {team_info}"
                team_uuid = team_info.get('id')
                if not team_uuid:
                    return "Could not find team UUID"

            query = """
                mutation CreateIssue($title: String!, $description: String!, $teamId: String!, $priority: Int, $stateId: String) {
                    issueCreate(input: {
                        title: $title,
                        description: $description,
                        teamId: $teamId,
                        priority: $priority,
                        stateId: $stateId
                    }) {
                        success
                        issue {
                            id
                            title
                            url
                            state {
                                id
                                name
                            }
                        }
                    }
                }
            """

            result = client._execute_query(query, {
                "title": title,
                "description": description,
                "teamId": team_uuid,
                "priority": priority,
                "stateId": state_id
            })
            return result["data"]["issueCreate"]
        except Exception as e:
            return f"Error creating issue: {str(e)}"
