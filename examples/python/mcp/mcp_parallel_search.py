"""Use free, keyless Parallel Search through Orchestra's existing MCP adapter."""

import argparse
import asyncio
import os
from typing import Callable, Optional

from mainframe_orchestra import Agent, MCPOrchestra, OpenaiModels, Task, __version__


async def run_parallel_search(query: str, llm: Optional[Callable] = None) -> str:
    """Return search excerpts, or let a supplied model answer using MCP tools."""
    # The MCP stdio client forwards a limited environment by default. Preserve
    # configured network settings without forwarding model or service API keys.
    bridge_env = {
        name: os.environ[name]
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "NODE_EXTRA_CA_CERTS",
            "MCP_REMOTE_CONFIG_DIR",
        )
        if name in os.environ
    }
    async with MCPOrchestra() as client:
        await client.connect(
            server_name="parallel_search",
            command="npx",
            args=[
                "-y",
                "mcp-remote@0.14.3",
                "https://search.parallel.ai/mcp",
                "--transport",
                "http-only",
                "--enable-proxy",
                "--header",
                f"User-Agent: mainframe-orchestra/{__version__}",
            ],
            env=bridge_env,
        )
        tools = client.get_server_tools("parallel_search")
        if llm is None:
            search = next(tool for tool in tools if tool.__name__ == "web_search")
            return await search(objective=query, search_queries=[query])

        agent = Agent(
            agent_id="parallel_researcher",
            role="Web Research Assistant",
            goal="Answer questions using web sources and cite their URLs",
            tools=tools,
            llm=llm,
        )
        return await Task.create_async(
            agent=agent,
            instruction=(
                f"Research this question: {query}\n"
                "Use web_search first and cite source URLs in your answer. "
                "Use web_fetch only if the search excerpts are insufficient."
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or search query")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Generate an answer with OpenAI (requires OPENAI_API_KEY)",
    )
    options = parser.parse_args()
    model = OpenaiModels.gpt_4o if options.agent else None
    print(asyncio.run(run_parallel_search(options.query, llm=model)))
