# Linear Tools

The LinearTools class provides methods for interacting with the Linear API, allowing you to manage issues, workflows, and team-related operations.

### Configuration

Before using Linear API operations, set up your environment variables:

```bash
export LINEAR_API_KEY=your_linear_api_key
export LINEAR_TEAM_ID=your_team_id
```

### Class Methods

##### get_team_issues(team_id: str = None, status: Optional[str] = None)

Lists issues for a specific team, optionally filtered by status.

```python
# Get all team issues
issues = LinearTools.get_team_issues()

# Get issues with specific status
issues = LinearTools.get_team_issues(status="In Progress")

# Get issues for a different team
issues = LinearTools.get_team_issues(team_id="TEAM_ID")
```

##### update_issue_status(issue_id: str, status_id: str)

Updates the status of an issue.

```python
result = LinearTools.update_issue_status("ISSUE_ID", "STATUS_ID")
```

##### search_issues(search_query: str)

Searches for issues using a text query.

```python
issues = LinearTools.search_issues("bug in authentication")
```

##### get_team_by_name(team_name: str)

Gets team information by team key/name. Also prints available teams for reference.

```python
team = LinearTools.get_team_by_name("ENG")
```

##### get_workflow_states(team_id: str = None)

Gets all workflow states for a team.

```python
states = LinearTools.get_workflow_states()
```

##### create_issue(title: str, description: str, team_id: str = None, priority: Optional[int] = None, state_id: Optional[str] = None)

Creates a new issue in Linear.

```python
issue = LinearTools.create_issue(
    title="Fix authentication bug",
    description="Users are unable to log in when...",
    priority=4,  # 4 is urgent
    state_id="STATE_ID"  # Optional initial state
)
```

### Return Data Structures

##### Issue Data
```python
{
    "id": "ISSUE_ID",
    "title": "Issue title",
    "state": {
        "name": "In Progress"
    },
    "priority": 2,
    "assignee": {
        "name": "John Doe"
    },
    "url": "https://linear.app/..."
}
```

##### Team Data
```python
{
    "id": "TEAM_ID",
    "name": "Engineering",
    "key": "ENG"
}
```

##### Workflow State Data
```python
{
    "id": "STATE_ID",
    "name": "In Progress",
    "type": "started"
}
```

### Usage Notes

1. **Authentication**: 
   - Requires `LINEAR_API_KEY` environment variable
   - Team operations require `LINEAR_TEAM_ID` environment variable
   - Both are validated on class initialization

2. **Error Handling**:
   - Methods return error messages as strings when operations fail
   - Successful operations return appropriate data structures
   - Environment variable missing errors are raised as ValueError

3. **Priority Levels**:
   - 0: No priority
   - 1: Low
   - 2: Medium
   - 3: High
   - 4: Urgent

4. **Team Management**:
   - Use `get_team_by_name()` to find team IDs
   - Team ID can be provided either as UUID or team key
   - Default team from `LINEAR_TEAM_ID` is used when team_id is not specified

5. **Issue Creation**:
   - Title and description are required
   - Priority and state_id are optional
   - Team ID defaults to environment variable if not provided
   - Returns issue data including URL and ID

6. **State Management**:
   - Use `get_workflow_states()` to find available states
   - State IDs are required for updating issue status
   - States are team-specific

The LinearTools class provides a streamlined interface to Linear's API while handling authentication, error checking, and response processing automatically. 