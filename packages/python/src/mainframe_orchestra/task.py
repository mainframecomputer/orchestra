# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from pydantic import BaseModel
from pydantic.fields import Field
from typing import Callable, Optional, Union, Dict, List, Any, Set, Tuple, AsyncIterator, Iterator
from datetime import datetime, date
import json
import re
import asyncio
import logging

# Configure logger for the orchestra package
logger = logging.getLogger("orchestra")


class LogColors:
    """ANSI color codes for log formatting"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    BLUE = "\033[1;34m"
    MAGENTA = "\033[1;35m"
    GREEN = "\033[32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    GREY = "\033[90m"

    @classmethod
    def wrap(cls, color: str, text: str) -> str:
        """Wrap text with color and reset codes"""
        return f"{color}{text}{cls.RESET}"


def parse_json_response(response: str) -> dict:
    """
    Parse a JSON response, handling potential formatting issues.

    Args:
        response (str): The JSON response string to parse.

    Returns:
        dict: The parsed JSON data.

    Raises:
        ValueError: If the JSON cannot be parsed after multiple attempts.
    """
    # Clean any markdown code blocks
    if response.startswith("```") and response.endswith("```"):
        # Split by newlines and remove first and last lines (```language and ```)
        lines = response.split("\n")
        if len(lines) > 2:
            response = "\n".join(lines[1:-1])

    try:
        # Try to parse the entire response
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}")

        # Find the first complete JSON object
        json_pattern = r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})"
        json_matches = re.finditer(json_pattern, response, re.DOTALL)

        for match in json_matches:
            try:
                result = json.loads(match.group(1))
                # Validate it's a dict and has expected structure
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        # Cleave strings before and after JSON
        cleaved_json = response.strip().lstrip("`").rstrip("`")
        try:
            return json.loads(cleaved_json)
        except json.JSONDecodeError as e:
            # Try removing any comments before parsing
            try:
                # Remove both single-line and multi-line comments
                comment_pattern = r"//.*?(?:\n|$)|/\*.*?\*/"
                cleaned_json = re.sub(comment_pattern, "", cleaved_json, flags=re.DOTALL)
                return json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                logger.error(f"All JSON parsing attempts failed: {e}")
                raise ValueError(f"Invalid JSON structure: {e}")


def serialize_result(obj: Any) -> Union[str, Dict[str, Any], List[Any]]:
    """Convert any object into a JSON-serializable format by aggressively stringifying non-standard types."""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            try:
                return {str(k): serialize_result(v) for k, v in obj.items()}
            except Exception as e:
                logger.warning(f"Failed to serialize dictionary: {e}")
                return str(obj)
        elif type(obj) in (list, tuple, set):  # Exact type checking for sequences
            try:
                return [serialize_result(item) for item in obj]
            except Exception as e:
                logger.warning(f"Failed to serialize sequence: {e}")
                return str(obj)
        elif hasattr(obj, "to_dict"):  # Handle objects with to_dict method
            try:
                return serialize_result(obj.to_dict())
            except Exception as e:
                logger.warning(f"Failed to serialize using to_dict: {e}")
                return str(obj)
        else:
            return str(obj)  # Fallback for all other types
    except Exception as e:
        logger.error(f"Serialization failed, using str() fallback: {e}")
        try:
            return str(obj)
        except Exception as e:
            logger.error(f"str() fallback failed: {e}")
            return f"<Unserializable object of type {type(obj).__name__}>"


def print_event(event: Dict[str, Any]) -> None:
    """Pretty print events with color coding."""
    if event["type"] == "tool_call" and "summary" in event:
        logger.info(f"Tool call: {event['summary']}")
        print(f"\n{LogColors.MAGENTA}🔧 {event['summary']}{LogColors.RESET}\n")
    elif event["type"] == "tool_result":
        logger.info(f"Tool result: {event['result']}")
        print(f"{LogColors.GREEN}Result: {event['result']}{LogColors.RESET}\n")
    elif event["type"] == "error":
        logger.error(event["content"])
        print(f"{LogColors.RED}Error: {event['content']}{LogColors.RESET}\n")
    elif event["type"] == "warning":
        logger.warning(event["content"])
        print(f"{LogColors.YELLOW}Warning: {event['content']}{LogColors.RESET}\n")
    elif event["type"] == "stream":
        print(event["content"], end="", flush=True)
    elif event["type"] == "initial_response":
        if event.get("streaming"):
            print(event["content"], end="", flush=True)
        else:
            logger.info("Initial response provided")
            print(f"\n{LogColors.CYAN}Initial Response: {LogColors.RESET}{event['content']}\n")
    elif event["type"] == "final_response":
        if event.get("streaming"):
            print(event["content"], end="", flush=True)
        else:
            logger.info("Final response provided")
            print(f"\n{LogColors.CYAN}Final Response: {LogColors.RESET}{event['content']}\n")
    elif event["type"] == "tool_status":
        logger.info(event["content"])
        print(f"\n{LogColors.CYAN}{event['content']}{LogColors.RESET}\n")
    elif event["type"] == "fallback_attempt":
        logger.warning(event["content"])
        print(f"\n{LogColors.YELLOW}{event['content']}{LogColors.RESET}\n")
    elif event["type"] == "end_tool_use":
        logger.info("Tool loop completed")
        print(f"\n{LogColors.CYAN}Tool Use: {LogColors.BLUE}Complete{LogColors.RESET}\n")


class Task(BaseModel):
    """
    Represents a task to be executed by an agent.

    Attributes:
        agent_id (Optional[str]): The ID of the agent performing the task
        role (str): The role or type of agent performing the task
        goal (str): The objective or purpose of the task
        attributes (Optional[str]): Additional attributes of the agent
        context (Optional[str]): Background information for the task
        instruction (str): Specific directions for completing the task
        llm (Union[Callable, List[Callable], Tuple[Callable, ...]]): Language model function(s)
        tools (Optional[Set[Callable]]): Optional set of tool functions
        image_data (Optional[Union[List[str], str]]): Optional base64-encoded image data
        temperature (Optional[float]): Temperature setting for the LLM (default: 0.7)
        max_tokens (Optional[int]): Maximum tokens for LLM response (default: 4000)
        require_json_output (bool): Whether to request JSON output
        stream (bool): Whether to stream the final LLM response
        initial_response (bool): Whether to provide initial response before tools
        tool_summaries (bool): Whether to include summaries for tool calls
    """

    # Agent-specific fields
    agent_id: Optional[str] = Field(None, description="The ID of the agent performing the task")
    role: str = Field(..., description="The role or type of agent performing the task")
    goal: str = Field(..., description="The objective or purpose of the task")
    attributes: Optional[str] = Field(None, description="Additional attributes or characteristics of the agent or expected responses")
    agent: Optional[Any] = Field(None, description="The agent associated with this task")

    # Core task inputs
    instruction: str = Field(..., description="Specific directions for completing the task")
    context: Optional[str] = Field(None, description="The background information or setting for the task")

    # Model configuration
    llm: Union[Callable, List[Callable], Tuple[Callable, ...]] = Field(..., description="The language model function(s) to be called. Can be a single function or multiple functions for fallback.")
    temperature: Optional[float] = Field(default=0.7, description="Temperature setting for the language model")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum number of tokens for the language model response")
    require_json_output: bool = Field(default=False, description="Whether to request JSON output from the LLM")

    # Input/Output handling
    image_data: Optional[Union[List[str], str]] = Field(None, description="Optional base64-encoded image data")
    stream: bool = Field(default=False, description="Whether to stream the final LLM response")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="List of messages in OpenAI chat format")

    # Tool configuration
    tools: Optional[Set[Callable]] = Field(default=None, description="Optional set of tool functions")
    tool_summaries: bool = Field(default=False, description="Whether to include explanatory summaries for tool calls")

    # Response handling
    initial_response: bool = Field(default=False, description="Whether to provide an initial response before tool execution")

    # Execution control
    thread_id: Optional[str] = Field(None, description="Thread ID for tracking conversation context")
    event_queue: Optional[Any] = Field(None, description="An optional event queue for inter-thread communication.")
    pre_execute: Optional[Callable[[Dict[str, Any]], None]] = Field(None, description="Optional pre-execution callback")

    # Pydantic configuration
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create(
        cls,
        agent: Optional[Any] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        attributes: Optional[str] = None,
        context: Optional[str] = None,
        instruction: Optional[str] = None,
        llm: Optional[Union[Callable, List[Callable], Tuple[Callable, ...]]] = None,
        tools: Optional[Set[Callable]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json_output: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_queue: Optional[Any] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        initial_response: bool = False,
        tool_summaries: bool = False,
    ) -> Union[str, Exception, AsyncIterator[str]]:
        """Create and execute a task. Handles both sync and async execution."""

        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("[Task.create] Created new event loop")

            # If we're already in an async context, return the coroutine
            if loop.is_running():
                return cls._create_async(
                    agent,
                    role,
                    goal,
                    attributes,
                    context,
                    instruction,
                    llm,
                    tools,
                    image_data,
                    temperature,
                    max_tokens,
                    require_json_output,
                    callback,
                    event_queue,
                    messages,
                    stream,
                    initial_response,
                    tool_summaries=tool_summaries,
                )

            # Otherwise, run it synchronously
            result = loop.run_until_complete(
                cls._create_async(
                    agent,
                    role,
                    goal,
                    attributes,
                    context,
                    instruction,
                    llm,
                    tools,
                    image_data,
                    temperature,
                    max_tokens,
                    require_json_output,
                    callback,
                    event_queue,
                    messages,
                    stream,
                    initial_response,
                    tool_summaries=tool_summaries,
                )
            )
            return result
        except Exception as e:
            print(f"[Task.create] Error during task creation: {str(e)}")
            return e

    @classmethod
    async def _create_async(
        cls,
        agent: Optional[Any] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        attributes: Optional[str] = None,
        context: Optional[str] = None,
        instruction: Optional[str] = None,
        llm: Optional[Union[Callable, List[Callable], Tuple[Callable, ...]]] = None,
        tools: Optional[Set[Callable]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json_output: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_queue: Optional[Any] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        initial_response: bool = False,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
        thread_id: Optional[str] = None,
        tool_summaries: bool = False,
    ) -> Union[str, Exception, AsyncIterator[str]]:
        """
        Create and execute a task asynchronously.

        Args:
            agent: Optional agent instance
            role: Role or type of agent
            goal: Task objective
            attributes: Additional agent attributes
            context: Task background information
            instruction: Task directions
            llm: Language model function(s)
            tools: Optional tool functions
            image_data: Optional image data
            temperature: LLM temperature
            max_tokens: Maximum tokens
            require_json_output: Whether to request JSON
            callback: Optional progress callback
            event_queue: Optional event queue
            messages: Optional message history
            stream: Whether to stream response
            initial_response: Whether to provide initial response
            pre_execute: Optional pre-execution callback
            thread_id: Optional thread ID
            tool_summaries: Whether to include tool summaries

        Returns:
            Union[str, AsyncIterator[str]]: Task result

        Raises:
            ValueError: If required parameters are missing
            Exception: If task execution fails
        """
        try:
            # Validate required parameters
            if not role and not (agent and agent.role):
                raise ValueError("Role must be provided either directly or via agent")
            if not goal and not (agent and agent.goal):
                raise ValueError("Goal must be provided either directly or via agent")
            if not instruction:
                raise ValueError("Instruction is required")
            if not llm and not (agent and agent.llm):
                raise ValueError("LLM function must be provided either directly or via agent")

            messages = messages or []
            if not messages or messages[0].get("role") != "system":
                system_message = {
                    "role": "system",
                    "content": (
                        f"You are {role or (agent.role if agent else None)}. "
                        f"Your goal is {goal or (agent.goal if agent else None)}"
                        f"{' Your attributes are: ' + (attributes or (agent.attributes if agent else '')) if attributes or (agent.attributes if agent else '') else ''}"
                    ).strip(),
                }
                messages.insert(0, system_message)

            # Combine context and instruction
            user_content = []
            if context:
                user_content.append(context)
            user_content.append(instruction)
            user_message_content = "\n\n".join(user_content)

            # Only append if different from last message
            if not messages or messages[-1].get("content") != user_message_content:
                messages.append({"role": "user", "content": user_message_content})

            task_data = {
                "agent_id": agent.agent_id if agent else None,
                "role": role or (agent.role if agent else None),
                "goal": goal or (agent.goal if agent else None),
                "attributes": attributes or (agent.attributes if agent else None),
                "context": context,
                "instruction": instruction,
                "llm": llm or (agent.llm if agent else None),
                "tools": tools
                or getattr(agent, "tools", None)
                or None,  # Handle missing tools attribute
                "image_data": image_data,
                "temperature": temperature or (agent.temperature if agent else 0.7),
                "max_tokens": max_tokens or (agent.max_tokens if agent else 4000),
                "require_json_output": require_json_output,
                "agent": agent,
                "event_queue": event_queue,
                "messages": messages,
                "stream": stream,
                "pre_execute": pre_execute,
                "initial_response": initial_response,
                "tool_summaries": tool_summaries,
            }

            # Validate task data using Pydantic
            task = cls.model_validate(task_data)

            logger.info(f"Created task for agent {task.agent_id or 'unknown'}")
            return await task.execute(callback, pre_execute)

        except Exception as e:
            error_msg = f"Failed to create task: {str(e)}"
            logger.error(error_msg)
            if callback:
                await callback({"type": "error", "content": error_msg, "thread_id": thread_id})
            raise

    async def execute(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Execute the task with optional tool usage.

        Args:
            callback: Optional progress callback function
            pre_execute: Optional pre-execution callback

        Returns:
            Union[str, AsyncIterator[str]]: Task execution result

        Raises:
            Exception: If task execution fails
        """
        try:
            if pre_execute:
                await pre_execute({"agent_id": self.agent_id})

            logger.info(f"Executing task for agent {self.agent_id or 'unknown'}")

            if self.tools:
                tool_result, tool_history = await self._execute_tool_loop(callback, pre_execute)
                if isinstance(tool_result, Exception):
                    raise tool_result
                return await self._execute_final_task(tool_history, callback)
            else:
                return await self._direct_llm_call(callback)

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)
            if callback:
                await callback({"type": "error", "content": error_msg, "agent_id": self.agent_id})
            raise

    async def _direct_llm_call(
        self, callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, AsyncIterator[str]]:
        """Execute a direct LLM call without tool usage, with fallback support."""
        logger = logging.getLogger("orchestra")

        # Log messages in a single line
        logger.info("[LLM Request] Messages: " + json.dumps(self.messages, separators=(",", ":")))

        # Define llm_params before logging
        llm_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "image_data": self.image_data,
            "require_json_output": self.require_json_output,
            "stream": self.stream,
        }

        # Log parameters in a single line
        logger.debug("[LLM Params] " + json.dumps(llm_params, separators=(",", ":")))

        # Convert single LLM to list for unified handling
        llms = [self.llm] if callable(self.llm) else list(self.llm)
        last_error = None

        for i, llm in enumerate(llms, 1):
            try:
                if self.stream:

                    async def stream_wrapper():
                        async for chunk in await llm(messages=self.messages, **llm_params):
                            if callback:
                                await callback(
                                    {
                                        "type": "stream",
                                        "content": chunk,
                                        "agent_id": self.agent_id,
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                )
                            yield chunk

                    return stream_wrapper()

                # Non-streaming response
                if callback and len(llms) > 1:
                    await callback(
                        {
                            "type": "fallback_attempt",
                            "content": f"Attempting LLM {i}/{len(llms)}",
                            "agent_id": self.agent_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                llm_result = await llm(messages=self.messages, **llm_params)

                if isinstance(llm_result, tuple) and len(llm_result) == 2:
                    response, error = llm_result
                    if error:
                        raise error
                elif isinstance(llm_result, dict):
                    response = json.dumps(llm_result)
                elif isinstance(llm_result, str):
                    response = llm_result
                else:
                    raise ValueError(f"Unexpected result type from LLM: {type(llm_result)}")

                if callback:
                    await callback(
                        {
                            "type": "final_response",
                            "content": response,
                            "agent_id": self.agent_id,
                            "timestamp": datetime.now().isoformat(),
                            "attempt": i if len(llms) > 1 else None,
                        }
                    )
                return response.strip()

            except Exception as e:
                last_error = e
                logger.error(f"LLM attempt {i}/{len(llms)} failed: {str(e)}")
                if callback:
                    await callback(
                        {
                            "type": "error",
                            "content": f"LLM attempt {i}/{len(llms)} failed: {str(e)}",
                            "agent_id": self.agent_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                if i < len(llms):
                    continue
                raise last_error

    async def _execute_tool_loop(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Union[str, Exception], List[str]]:
        """Execute the tool loop with enhanced logging."""
        logger = logging.getLogger("orchestra")
        logger.info("Starting tool loop execution")

        try:
            MAX_ITERATIONS = 10
            MAX_IDENTICAL_CALLS = 3
            iteration_count = 0
            tool_call_history = {}
            tool_results = []  # Simple list to store results

            def hash_tool_call(tool_call: dict) -> str:
                """Create a hash of a tool call to detect duplicates."""
                tool_str = f"{tool_call.get('tool')}_{json.dumps(tool_call.get('params', {}), sort_keys=True)}"
                return tool_str

            tool_descriptions = (
                "\nAvailable Tools:\n"
                + "\n".join([f"- {func.__name__}: {func.__doc__}" for func in self.tools]).rstrip()
            )

            more = "more " if len(self.tools) > 1 else ""
            additional = "additional " if len(self.tools) > 1 else ""

            tool_call_format_basic = """{
    "tool_calls": [
        {
            "tool": "tool_name",
            "params": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    ]
}"""

            tool_call_format_with_summary = """{
    "tool_calls": [
        {
            "tool": "tool_name",
            "params": {
                "param1": "value1",
                "param2": "value2"
            },
            "summary": "Brief explanation in active, present tense (e.g., 'Creating a new file') of why this tool is being called"
        }
    ]
}"""

            no_tools_format = """{
    "tool_calls": []
}
IMPORTANT: When indicating no more tools are needed, return ONLY the above JSON with no additional text or explanation."""

            tool_call_format = (
                tool_call_format_with_summary if self.tool_summaries else tool_call_format_basic
            )

            tool_loop_prompt = f"""
You are now determining if you need to call {more}tools to gather {more}information or perform {additional}actions to complete the given task, or if you are done using tools and are ready to proceed to the final response. Use your tools with persistence and patience to get the best results, and retry if you get a fixable error.

If you need to make tool calls, consider whether to make them successively or all at once. If the result of one tool call is required as input for another tool, make your calls one at a time. If multiple tool calls can be made independently, you may request them all at once.

Now respond with a JSON object in one of these formats:

If tool calls are still needed:
{tool_call_format}

If no more tool calls are required:
{no_tools_format}

Now respond with a JSON object that either requests tool calls or exits the tool loop. Do not comment before or after the JSON, and do not include any backticks or language declarations. Return only a valid JSON in any case.
"""

            while iteration_count < MAX_ITERATIONS:
                logger.info(f"Starting iteration {iteration_count + 1}/{MAX_ITERATIONS}")
                iteration_count += 1

                # Include tool results in context if we have any
                context_parts = []
                if self.context:
                    context_parts.append(self.context)
                context_parts.append(tool_descriptions)

                if tool_results:  # Using existing tool_results list
                    context_parts.append(
                        "**Tool Execution History:**\n"
                        "You have already performed these actions:\n\n"
                        + "\n".join(
                            [
                                f"#{i+1}. {result.strip()}"  # Add numbering
                                for i, result in enumerate(tool_results)
                            ]
                        )
                        + "\n\nReview these results before making new tool calls. Avoid repeating the same calls."
                    )

                tool_context = "\n-----\n".join(context_parts).strip()

                tool_loop_instruction = f"""
{tool_context}

=====
The original task instruction:
{self.instruction}
=====

{tool_loop_prompt}
"""

                temp_history = self.messages.copy()

                temp_history.append({"role": "user", "content": tool_loop_instruction})

                response, error = await self.llm(
                    messages=temp_history, require_json_output=True, temperature=self.temperature
                )

                if error:
                    logger.error(f"Error from LLM: {error}")
                    if callback:
                        await callback({"type": "error", "content": str(error)})
                    return error, []

                try:
                    response_data = parse_json_response(response)

                    # Validate basic response structure
                    if not isinstance(response_data, dict):
                        raise ValueError("Response must be a JSON object")

                    if "tool_calls" not in response_data:
                        raise ValueError("Response must contain 'tool_calls' key")

                    if not isinstance(response_data["tool_calls"], list):
                        raise ValueError("'tool_calls' must be an array")

                    # Handle explicit completion
                    if len(response_data["tool_calls"]) == 0:
                        logger.info("Received explicit completion signal (empty tool_calls)")
                        print(f"\n{LogColors.CYAN}Tool Use: {LogColors.BLUE}Loop Exited{LogColors.RESET}\n")
                        if callback:
                            await callback(
                                {
                                    "type": "end_tool_use",
                                    "content": "Tool usage complete",
                                    "agent_id": self.agent_id,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                        return None, tool_results

                    # Validate each tool call before proceeding
                    tools_dict = {func.__name__: func for func in self.tools}
                    for tool_call in response_data["tool_calls"]:
                        if not isinstance(tool_call, dict):
                            raise ValueError("Each tool call must be an object")

                        if "tool" not in tool_call:
                            raise ValueError("Each tool call must specify a 'tool' name")

                        if "params" not in tool_call:
                            raise ValueError("Each tool call must include 'params'")

                        if not isinstance(tool_call["params"], dict):
                            raise ValueError("Tool 'params' must be an object")

                        tool_name = tool_call.get("tool")
                        if tool_name not in tools_dict and tool_name != "conduct_tool":
                            raise ValueError(
                                f"Unknown tool: {tool_name}. Available tools: {', '.join(tools_dict.keys())}"
                            )

                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Invalid tool response: {str(e)}"
                    logger.error(f"{LogColors.RED}[TOOL_LOOP] {error_msg}{LogColors.RESET}")
                    logger.error(
                        f"Problematic response: {response[:200]}..."
                    )  # Log truncated response

                    if callback:
                        await callback(
                            {
                                "type": "error",
                                "content": error_msg,
                                "response": response[:1000],  # Truncate very long responses
                                "iteration": iteration_count,
                            }
                        )

                    # Add error to tool results for context in next iteration
                    tool_results.append(
                        f"\nTool Response Error:\n"
                        f"Iteration: {iteration_count}\n"
                        f"Error: {error_msg}\n"
                        f"Response: {response[:200]}..."  # Truncate long responses
                    )

                    # Continue to next iteration
                    continue

                # If we get here, all tool calls are valid - proceed with execution
                if "tool_calls" in response_data:
                    # Check initial_response flag BEFORE executing any tools
                    if self.initial_response and iteration_count <= 1:
                        logger.info("[TOOL_LOOP] Preparing initial response before executing tools")

                        initial_prompt = (
                            f"Given the task instruction: '{self.instruction}' and the planned tool calls: {json.dumps(response_data['tool_calls'], indent=2)}, "
                            "please provide an initial response explaining your planned approach before executing the tools."
                        )

                        original_messages = self.messages
                        original_json_requirement = self.require_json_output
                        self.messages = self.messages.copy()
                        self.messages.append({"role": "user", "content": initial_prompt})
                        self.require_json_output = False

                        try:
                            if self.stream:
                                # Always provide a callback wrapper, even if the original callback is None
                                async def callback_wrapper(event):
                                    print(f"{event.get('content', '')}", end="", flush=True)
                                    if callback:
                                        await callback(
                                            {**event, "type": "initial_response", "streaming": True}
                                        )

                                initial_stream = await self._direct_llm_call(callback_wrapper)
                                # Consume the stream
                                async for chunk in initial_stream:
                                    pass  # The callback will handle the chunks
                        finally:
                            self.require_json_output = original_json_requirement
                            self.messages = original_messages

                    # Now proceed with tool execution
                    for tool_call in response_data["tool_calls"]:
                        logger.info(f"Processing tool call: {tool_call.get('tool')}")
                        if isinstance(tool_call, dict):
                            tool_call_hash = hash_tool_call(tool_call)
                            call_count = tool_call_history.get(tool_call_hash, 0) + 1
                            tool_call_history[tool_call_hash] = call_count

                            logger.debug(f"Tool call count for this configuration: {call_count}")

                            if call_count > MAX_IDENTICAL_CALLS:
                                warning_msg = (
                                    f"Exiting tool loop due to verbatim repetition (suggesting infinite loop). "
                                    f"Tool '{tool_call.get('tool')}' with parameters {tool_call.get('params')} "
                                    f"has been called {call_count} times. Maximum allowed repetitions is {MAX_IDENTICAL_CALLS}."
                                )
                                logger.warning(f"{LogColors.YELLOW}{warning_msg}{LogColors.RESET}")
                                if callback:
                                    await callback({"type": "warning", "content": warning_msg})
                                # Instead of returning an error, return None to proceed to final task
                                return None, tool_results

                            if "task_id" in tool_call:
                                tool_name = "conduct_tool"
                                tool_params = {"instruction": [tool_call]}
                            else:
                                tool_name = tool_call.get("tool")
                                tool_params = tool_call.get("params", {})

                            logger.info(
                                f"Executing tool: {tool_name} with parameters: {json.dumps(tool_params, separators=(',', ':'))}"
                            )

                            # Send tool call event
                            if callback:
                                callback_data = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "params": tool_params,
                                    "agent_id": self.agent_id,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                # Add summary if available and tool_summaries is enabled
                                if self.tool_summaries and "summary" in tool_call:
                                    callback_data["summary"] = tool_call["summary"]
                                await callback(callback_data)

                            # Add colored output for tool call
                            print(f"\n{LogColors.CYAN}Tool Use: {LogColors.BLUE}{tool_name}")
                            # Add summary to console output
                            if self.tool_summaries and "summary" in tool_call:
                                print(
                                    f"{LogColors.CYAN}Summary: {LogColors.MAGENTA}{tool_call['summary']}"
                                )
                            print(f"{LogColors.CYAN}Parameters:")
                            for key, value in tool_params.items():
                                print(f"  {LogColors.CYAN}{key}: {LogColors.MAGENTA}{value}")
                            print(f"{LogColors.RESET}")  # Reset color at the end

                            # Execute tool and store result
                            tools_dict = {func.__name__: func for func in self.tools}
                            if tool_name not in tools_dict:
                                error_msg = f"Unknown tool: {tool_name}"
                                logger.error(f"{LogColors.RED}{error_msg}{LogColors.RESET}")
                                if callback:
                                    await callback({"type": "error", "content": error_msg})
                                return Exception(error_msg), []

                            try:
                                tool_func = tools_dict[tool_name]

                                if tool_name == "conduct_tool":
                                    logger.info(
                                        "[TOOL_LOOP] Setting up conduct_tool specific parameters"
                                    )
                                    tool_params["callback"] = callback
                                    tool_params["thread_id"] = self.thread_id
                                    tool_params["event_queue"] = self.event_queue
                                    tool_params["pre_execute"] = pre_execute

                                if asyncio.iscoroutinefunction(tool_func):
                                    logger.info(
                                        f"{LogColors.CYAN}Executing async tool: {tool_name}{LogColors.RESET}"
                                    )
                                    raw_result = await tool_func(**tool_params)
                                else:
                                    raw_result = tool_func(**tool_params)

                                # Check if the result is an exception
                                if isinstance(raw_result, Exception):
                                    error_msg = f"Tool returned error: {str(raw_result)}"
                                    formatted_error = (
                                        f"\nTool Execution Result:\n"
                                        f"Tool: '{tool_name}'\n"
                                        f"Parameters: {json.dumps(tool_params, indent=2)}\n"
                                        f"Error: {str(raw_result)}"
                                    )
                                    tool_results.append(formatted_error)
                                    # Continue execution to let the agent handle the error
                                    continue

                                # Process successful result as before
                                result = serialize_result(raw_result)

                                # Convert to string for message history if needed
                                result_str = (
                                    json.dumps(result, indent=2)
                                    if isinstance(result, (dict, list))
                                    else str(result)
                                )

                                # Add result snippet output
                                result_snippet = (
                                    result_str[:400] + "..."
                                    if len(result_str) > 400
                                    else result_str
                                )
                                print(
                                    f"{LogColors.CYAN}Result: {LogColors.GREEN}{result_snippet}{LogColors.RESET}"
                                )
                                print()

                                if callback:
                                    await callback(
                                        {
                                            "type": "tool_result",
                                            "tool": tool_name,
                                            "result": result_str,
                                            "agent_id": self.agent_id,
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    )

                                formatted_result = (
                                    f"\nTool Execution:\n"
                                    f"Tool: '{tool_name}'\n"
                                    f"Parameters: {json.dumps(tool_params, indent=2)}\n"
                                    f"Result:\n{result_str}"
                                )
                                tool_results.append(
                                    formatted_result
                                )  # Using existing tool_results list

                                # After tool execution
                                logger.info(
                                    f"Result from '{tool_name}': {json.dumps(result, separators=(',', ':'))}"
                                )

                            except Exception as e:
                                error_msg = f"Tool execution error for {tool_name}: {str(e)}"
                                logger.error(
                                    f"{LogColors.RED}[TOOL_LOOP] {error_msg}{LogColors.RESET}"
                                )
                                if callback:
                                    await callback({"type": "error", "content": error_msg})

                                # Format the error as a tool result
                                formatted_error = (
                                    f"\nTool Execution Error:\n"
                                    f"Tool: '{tool_name}'\n"
                                    f"Parameters: {json.dumps(tool_params, indent=2)}\n"
                                    f"Error: {str(e)}"
                                )
                                tool_results.append(formatted_error)

                                # Continue to the next iteration instead of returning
                                continue

                else:
                    logger.info("[TOOL_LOOP] No tool calls found in response")
                    return None, tool_results

            logger.info(f"Maximum iterations ({MAX_ITERATIONS}) reached")
            # Check for max iterations reached
            if iteration_count >= MAX_ITERATIONS:
                error_msg = f"Maximum tool loop iterations ({MAX_ITERATIONS}) reached"
                logger.error(f"{LogColors.RED}[TOOL_LOOP] {error_msg}{LogColors.RESET}")
                if callback:
                    await callback({"type": "error", "content": error_msg})
                return Exception(error_msg), tool_results

        except Exception as e:
            error_msg = f"Error in tool loop: {str(e)}"
            logger.error(f"{LogColors.RED}{error_msg}{LogColors.RESET}")
            if callback:
                await callback({"type": "error", "content": error_msg})
            return e, []

    async def _execute_final_task(
        self, tool_results: List[str], callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict, Exception, AsyncIterator[str]]:
        """Execute the final task with tool results."""
        logger = logging.getLogger("orchestra")
        logger.info("Starting final task execution")

        # Log tool results in a single line
        for idx, result in enumerate(tool_results):
            logger.info(f"[Tool Result {idx+1}] " + json.dumps(result, separators=(",", ":")))

        # Build content based on whether we have tool results
        content_parts = []

        if tool_results:
            content_parts.extend(
                [
                    "\nPrevious Tool Usage:",
                    "".join(tool_results),
                    "\nYou have just completed and exited your tool-use phase, and you are now writing your final response. Do not make any more tool calls.",
                ]
            )

        content_parts.append(f"Now focus on addressing the instruction:\n{self.instruction}")

        self.messages.append({"role": "user", "content": "\n".join(content_parts)})

        try:
            llm_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "image_data": self.image_data,
                "stream": self.stream,
            }

            if self.require_json_output:
                llm_params["require_json_output"] = True

            # Use the existing _direct_llm_call method which handles fallbacks
            result = await self._direct_llm_call(callback)

            if isinstance(result, Exception):
                return result

            if self.require_json_output and not isinstance(result, (AsyncIterator, Iterator)):
                try:
                    if isinstance(result, str):
                        json_match = re.search(r"\{.*?\}", result, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(0))
                    return json.loads(result)

                except json.JSONDecodeError as e:
                    return ValueError(f"Failed to parse JSON from LLM response: {result}\nError: {e}")


            return result

        except Exception as e:
            logger.error(f"[FINAL_TASK] Error in final task execution: {str(e)}")
            if callback:
                await callback({"type": "error", "content": str(e)})
            return e

    @staticmethod
    def process_stream(
        stream: AsyncIterator[str],
        callback: Optional[Callable[[str], Any]] = print,
        end: str = "",
        flush: bool = True,
    ) -> str:
        """Process a stream of text chunks, optionally collecting them.

        Args:
            stream (AsyncIterator[str]): The text stream to process
            callback (Optional[Callable]): Function to process each chunk. Defaults to print.
                If None, chunks are only collected without processing.
            end (str): String to append after each chunk when using print callback
            flush (bool): Whether to flush after each chunk when using print callback

        Returns:
            str: The complete concatenated text from all chunks
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def process():
            collected = []
            async for chunk in stream:
                if isinstance(chunk, dict):
                    # Handle streaming responses (both initial and final)
                    if chunk.get("type") in ["initial_response", "final_response"] and chunk.get(
                        "streaming"
                    ):
                        content = chunk["content"]
                        collected.append(content)
                        if callback:
                            if callback == print:
                                callback(content, end=end, flush=flush)
                            else:
                                callback(content)
                    # Handle non-streaming responses
                    elif chunk.get("type") in ["initial_response", "final_response"]:
                        content = chunk["content"]
                        collected.append(content)
                        if callback:
                            if callback == print:
                                callback(
                                    f"\n{chunk['type'].replace('_', ' ').title()}: {content}",
                                    end=end,
                                    flush=flush,
                                )
                            else:
                                callback(content)
                else:
                    # Handle direct string chunks
                    collected.append(chunk)
                    if callback:
                        if callback == print:
                            callback(chunk, end=end, flush=flush)
                        else:
                            callback(chunk)
            return "".join(collected)

        return loop.run_until_complete(process())


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for the orchestra package"""
    logging_level = getattr(logging, level.upper())

    # Create logger
    logger = logging.getLogger("orchestra")
    logger.setLevel(logging_level)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatter with more detailed format
    detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    formatter = logging.Formatter(detailed_format)

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Log initial configuration
    logger.info(f"Logging configured: level={level}, file={log_file}")
    logger.debug("Debug logging is enabled")

    # Log Python version and environment info
    import sys

    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")


def default_logger(event: Dict[str, Any]) -> None:
    """Default logging callback handler for Task events with colored output."""
    logger = logging.getLogger("orchestra")

    # Log the raw event at debug level
    logger.debug(f"Received event: {json.dumps(event, indent=2)}")

    if event["type"] == "tool_call":
        summary = event.get("summary", "No summary provided")
        logger.info(f"Tool Call - {event['tool']}: {summary}")
        logger.debug(f"Tool parameters: {json.dumps(event.get('params', {}), indent=2)}")

        if "summary" in event:
            logger.info(f"Tool Call Summary: {event['summary']}")
            print(f"\n{LogColors.MAGENTA}Tool Call: {event['summary']}{LogColors.RESET}")
        logger.info(f"Tool Name: {event['tool']}")
        logger.info(f"Tool Parameters: {json.dumps(event.get('params', {}), indent=2)}")
        print(f"{LogColors.CYAN}Tool: {LogColors.BLUE}{event['tool']}")
        print(
            f"{LogColors.CYAN}Parameters: {LogColors.MAGENTA}{json.dumps(event.get('params', {}), indent=2)}{LogColors.RESET}"
        )

    elif event["type"] == "tool_result":
        logger.info(f"Tool Result: {event['result']}")  # Log entire tool result
        logger.debug(f"Full tool result: {event['result']}")
        print(f"{LogColors.GREEN}Result: {event['result']}{LogColors.RESET}\n")

    elif event["type"] == "error":
        logger.error(f"Error event: {event['content']}")
        print(f"{LogColors.RED}Error: {event['content']}{LogColors.RESET}\n")

    elif event["type"] == "warning":
        logger.warning(f"Warning event: {event['content']}")
        print(f"{LogColors.YELLOW}Warning: {event['content']}{LogColors.RESET}\n")

    elif event["type"] == "stream":
        logger.debug(f"Stream content: {event['content']}")
        print(event["content"], end="", flush=True)

    elif event["type"] == "initial_response":
        if event.get("streaming"):
            logger.debug(f"Streaming initial response: {event['content']}")
            print(event["content"], end="", flush=True)
        else:
            logger.info("Initial response provided")
            logger.debug(f"Initial response content: {event['content']}")
            print(f"\n{LogColors.CYAN}Initial Response: {LogColors.RESET}{event['content']}\n")

    elif event["type"] == "final_response":
        if event.get("streaming"):
            logger.debug(f"Streaming final response: {event['content']}")
            print(event["content"], end="", flush=True)
        else:
            logger.info("Final response provided")
            logger.debug(f"Final response content: {event['content']}")
            print(f"\n{LogColors.CYAN}Final Response: {LogColors.RESET}{event['content']}\n")

    elif event["type"] == "tool_status":
        logger.info(f"Tool status: {event['content']}")
        print(f"\n{LogColors.CYAN}{event['content']}{LogColors.RESET}\n")

    elif event["type"] == "fallback_attempt":
        logger.warning(f"Fallback attempt: {event['content']}")
        print(f"\n{LogColors.YELLOW}{event['content']}{LogColors.RESET}\n")

    elif event["type"] == "end_tool_use":
        logger.info("Tool loop completed")
        print(f"\n{LogColors.CYAN}Tool Use: {LogColors.BLUE}Complete{LogColors.RESET}\n")

    else:
        logger.warning(f"Unknown event type received: {event['type']}")
        logger.debug(f"Full unknown event: {json.dumps(event, indent=2)}")
