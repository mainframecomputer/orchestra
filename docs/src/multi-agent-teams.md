# Multi-Agent Teams

Mainframe-Orchestra enables the creation and orchestration of multi-agent teams to tackle complex tasks. This section explores the Agent and Task classes, their interaction, and how to design effective multi-agent workflows.

### Multi-Agent Workflows

Here's an example of a multi-agent flow in Mainframe-Orchestra:

```python
import asyncio
from mainframe_orchestra import Agent, Task, WebTools, WikipediaTools, AmadeusTools, OpenrouterModels, set_verbosity
from datetime import datetime

set_verbosity(1)

async def main():
    web_research_agent = Agent(
        role="web research agent",
        goal="search the web thoroughly for travel information",
        attributes="hardworking, diligent, thorough, comprehensive.",
        llm=OpenrouterModels.gpt_4o,
        tools=[WebTools.serper_search, WikipediaTools.search_articles, WikipediaTools.search_images]
    )

    travel_agent = Agent(
        role="travel agent",
        goal="assist the traveller with their request",
        attributes="friendly, hardworking, and comprehensive and extensive in reporting back to users",
        llm=OpenrouterModels.gpt_4o,
        tools=[AmadeusTools.search_flights, WebTools.serper_search, WebTools.get_weather_data]
    )

    # Define the taskflow

    async def research_destination(destination, interests):
        destination_report = await Task.create_async(
            agent=web_research_agent,
            context=f"User Destination: {destination}\nUser Interests: {interests}",
            instruction=f"Use your tools to search relevant information about the given destination: {destination}. Use wikipedia tools to search the destination's wikipedia page, as well as images of the destination. In your final answer you should write a comprehensive report about the destination with images embedded in markdown."
        )
        return destination_report

    async def research_events(destination, dates, interests):
        events_report = await Task.create_async(
            agent=web_research_agent,
            context=f"User's intended destination: {destination}\n\nUser's intended dates of travel: {dates}\nUser Interests: {interests}",
            instruction="Use your tools to research events in the given location for the given date span. Ensure your report is a comprehensive report on events in the area for that time period."
        )
        return events_report

    async def research_weather(destination, dates):
        current_date = datetime.now().strftime("%Y-%m-%d")
        weather_report = await Task.create_async(
            agent=travel_agent,
            context=f"Location: {destination}\nDates: {dates}\n(Current Date: {current_date})",
            instruction="Use your weather tool to search for weather information in the given dates and write a report on the weather for those dates. Do not be concerned about dates in the future; ** IF dates are more than 10 days away, user web search instead of weather tool. If the dates are within 10 days, use the weather tool. ** Always search for weather information regardless of the date you think it is."
        )
        return weather_report

    async def search_flights(current_location, destination, dates):
        flight_report = await Task.create_async(
            agent=travel_agent,
            context=f"Current Location: {current_location}\n\nDestination: {destination}\nDate Range: {dates}",
            instruction=f"Search for a lot of flights in the given date range to collect a bunch of options and return a report on the best options in your opinion, based on convenience and lowest price."
        )
        return flight_report

    async def write_travel_report(destination_report, events_report, weather_report, flight_report):
        travel_report = await Task.create_async(
            agent=travel_agent,
            context=f"Destination Report: {destination_report}\n--------\n\nEvents Report: {events_report}\n--------\n\nWeather Report: {weather_report}\n--------\n\nFlight Report: {flight_report}",
            instruction=f"Write a comprehensive travel plan and report given the information above. Ensure your report conveys all the detail in the given information, from flight options, to weather, to events, and image urls, etc. Preserve detail and write your report in extensive length."
        )
        return travel_report

    current_location = input("Enter current location: ")
    destination = input("Enter destination: ")
    dates = input("Enter dates: ")
    interests = input("Enter interests: ")

    destination_report = await research_destination(destination, interests)
    print(destination_report)
    events_report = await research_events(destination, dates, interests)
    print(events_report)
    weather_report = await research_weather(destination, dates)
    print(weather_report)
    flight_report = await search_flights(current_location, destination, dates)
    print(flight_report)
    travel_report = await write_travel_report(destination_report, events_report, weather_report, flight_report)
    print(travel_report)

if __name__ == "__main__":
    asyncio.run(main())
```

Each agent is equipped with specific tools from the WebTools, WikipediaTools, and AmadeusTools modules, allowing them to interact with the web, Wikipedia, and Amadeus effectively. The agents work in sequence, each building upon the work of the previous one, creating a sequential workflow.

Multi-agent workflows offer several benefits. They allow for specialization, with each agent focusing on specific tasks. The workflow becomes modular, divided into smaller, manageable parts. These workflows can scale to handle complex tasks by distributing work among agents. They're also flexible, allowing easy modification to meet changing needs.

### Agent Orchestration with Conduct Tool

Orchestra provides a powerful orchestration capability through the `Conduct` tool, which enables agents to dynamically delegate tasks to other specialized agents. This creates a hierarchical structure where a coordinator agent can manage and direct a team of specialized agents.

Here's an example of orchestration in action:

```python
import asyncio
from mainframe_orchestra import Agent, Task, Conduct, OpenaiModels, GitHubTools, WebTools, SearchTools

async def main():
    # Define specialized agents
    github_agent = Agent(
        agent_id="github_agent",
        role="GitHub Specialist",
        goal="Analyze GitHub repositories, issues, and code",
        attributes="Expert in code analysis, repository structure, and GitHub operations",
        llm=OpenaiModels.gpt_4o,
        tools=[
            GitHubTools.search_repositories,
            GitHubTools.get_directory_structure,
            GitHubTools.get_file_content,
            GitHubTools.get_issue_comments,
            GitHubTools.list_repo_issues
        ]
    )

    web_research_agent = Agent(
        agent_id="web_research_agent",
        role="Web Research Specialist",
        goal="Conduct thorough web research and gather information from various sources",
        attributes="Expert in web search, content analysis, and information synthesis",
        llm=OpenaiModels.gpt_4o,
        tools=[
            WebTools.serper_search,
            WebTools.scrape_url_with_serper,
            SearchTools.search_news
        ]
    )

    shopping_agent = Agent(
        agent_id="shopping_agent",
        role="Product Research Specialist",
        goal="Find and compare products, prices, and shopping options",
        attributes="Expert in product search, price comparison, and e-commerce analysis",
        llm=OpenaiModels.gpt_4o,
        tools=[
            SearchTools.search_shopping,
            WebTools.scrape_url_with_serper
        ]
    )

    # Create coordinator agent with Conduct tool
    coordinator = Agent(
        agent_id="coordinator",
        role="Team Coordinator",
        goal="Coordinate specialized agents to accomplish complex multi-step tasks",
        attributes="""Expert at breaking down complex tasks and delegating to the right specialists.
        You understand each agent's capabilities and can orchestrate them effectively to achieve the best results.""",
        llm=OpenaiModels.gpt_4o,
        tools=[Conduct.conduct_tool(github_agent, web_research_agent, shopping_agent)]
    )

    # Example orchestration task
    user_request = input("Enter your request (e.g., 'Research the best Python web frameworks and find GitHub repositories with examples'): ")

    print(f"\n🎯 Coordinating team to handle: {user_request}")
    print("The coordinator will analyze your request and delegate tasks to specialized agents...\n")

    # The coordinator will automatically:
    # 1. Analyze the user request
    # 2. Break it down into subtasks
    # 3. Assign each subtask to the most appropriate specialist agent
    # 4. Manage dependencies between tasks
    # 5. Synthesize the results into a comprehensive response

    result = await Task.create_async(
        agent=coordinator,
        instruction=f"""
        Please coordinate your team of specialists to thoroughly address this user request: {user_request}

        You have access to:
        - github_agent: For analyzing GitHub repositories, code, and issues
        - web_research_agent: For general web research and information gathering
        - shopping_agent: For product research and price comparisons

        Break down the request into appropriate subtasks, delegate to the right specialists,
        and provide a comprehensive final report that synthesizes all findings.
        """
    )

    print("\n" + "="*80)
    print("🎯 ORCHESTRATED TEAM RESULTS")
    print("="*80)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Key Orchestration Features

The `Conduct` tool provides several powerful capabilities:

#### Dynamic Task Decomposition
The coordinator agent automatically breaks down complex requests into smaller, manageable subtasks that can be handled by specialist agents.

#### Intelligent Agent Assignment
Based on the nature of each subtask, the coordinator selects the most appropriate specialist agent, considering their tools and expertise.

#### Dependency Management
The orchestration system handles dependencies between tasks, ensuring that agents receive the output from previous tasks when needed.

#### Data Flow Optimization
Instead of routing all information through the coordinator, the system allows direct data flow between dependent tasks, reducing bottlenecks.

#### Hierarchical Delegation
Agents with the conduct tool can themselves delegate to other agents, creating nested orchestration patterns for complex workflows.

### Orchestration Best Practices

When designing orchestrated multi-agent systems:

- **Clear Specialization**: Define agents with clear, non-overlapping areas of expertise
- **Comprehensive Tool Sets**: Ensure each specialist has all the tools needed for their domain
- **Detailed Instructions**: Provide the coordinator with clear guidance on when to use each specialist
- **Task Boundaries**: Design tasks that can be cleanly delegated and don't require excessive back-and-forth
- **Error Handling**: Implement robust error handling since orchestration involves multiple agents
- **Performance Monitoring**: Monitor the performance of orchestrated workflows to identify bottlenecks

### Workflow Design

When designing multi-agent workflows, consider the following steps:

- Break down the overall problem into distinct subtasks.
- Create specialized agents for each subtask, defining their roles, goals, and tools.
- Design a sequence of tasks that pass information between agents.
- Use the Task.create_async() method to execute tasks for each agent in the workflow.
- Carefully curate tool sets for each agent, providing only the tools necessary for their specific tasks.
- Use descriptive attributes for agents to guide their behavior and decision-making process.
- Ensure smooth information flow between tasks by structuring your workflow carefully.
- Monitor and analyze the performance of your multi-agent teams to identify areas for improvement.
