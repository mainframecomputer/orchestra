# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import asyncio
import json
import logging
import queue
import threading
from datetime import date, datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from .utils.logging_config import logger
from .utils.parse_json_response import parse_json_response


def serialize_result(obj: Any) -> Union[str, int, float, bool, None, Dict[str, Any], List[Any]]:
    """Convert any object into a JSON-serializable format by aggressively stringifying non-standard types."""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, range):
            return list(obj)
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
        max_iterations (int): Maximum number of tool loop iterations (default: 50)
    """

    # Agent-specific fields
    agent_id: Optional[str] = Field(None, description="The ID of the agent performing the task")
    role: str = Field(..., description="The role or type of agent performing the task")
    goal: str = Field(..., description="The objective or purpose of the task")
    attributes: Optional[str] = Field(
        None,
        description="Additional attributes or characteristics of the agent or expected responses",
    )
    agent: Optional[Any] = Field(None, description="The agent associated with this task")

    # Core task inputs
    instruction: str = Field(..., description="Specific directions for completing the task")
    context: Optional[str] = Field(
        None, description="The background information or setting for the task"
    )

    # Model configuration
    llm: Union[Callable, List[Callable], Tuple[Callable, ...]] = Field(
        ...,
        description="The language model function(s) to be called. Can be a single function or multiple functions for fallback.",
    )
    temperature: Optional[float] = Field(
        default=0.7, description="Temperature setting for the language model"
    )
    max_tokens: Optional[int] = Field(
        default=4000, description="Maximum number of tokens for the language model response"
    )
    require_json_output: bool = Field(
        default=False, description="Whether to request JSON output from the LLM"
    )

    # Input/Output handling
    image_data: Optional[Union[List[str], str]] = Field(
        None, description="Optional base64-encoded image data"
    )
    stream: bool = Field(default=False, description="Whether to stream the final LLM response")
    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of messages in OpenAI chat format"
    )

    # Tool configuration
    tools: Optional[Set[Callable]] = Field(
        default=None, description="Optional set of tool functions"
    )
    tool_summaries: bool = Field(
        default=False, description="Whether to include explanatory summaries for tool calls"
    )
    max_iterations: int = Field(default=50, description="Maximum number of tool loop iterations")

    # Response handling
    initial_response: bool = Field(
        default=False, description="Whether to provide an initial response before tool execution"
    )

    # Execution control
    thread_id: Optional[str] = Field(
        None, description="Thread ID for tracking conversation context"
    )
    event_queue: Optional[Any] = Field(
        None, description="An optional event queue for inter-thread communication."
    )
    pre_execute: Optional[Callable[[Dict[str, Any]], None]] = Field(
        None, description="Optional pre-execution callback"
    )

    # Pydantic configuration
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools: Optional[Set[Callable]]) -> Optional[Set[Callable]]:
        """Validate that all tools have docstrings."""
        if tools:
            for tool in tools:
                if not tool.__doc__ or not tool.__doc__.strip():
                    raise ValueError(
                        f"Tool '{tool.__name__}' is missing a docstring or has an empty docstring. All tools must have documentation."
                    )
        return tools

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
        callback: Optional[
            Callable[[Dict[str, Any]], Union[None, Coroutine[Any, Any, None]]]
        ] = None,
        event_queue: Optional[Any] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        initial_response: bool = False,
        tool_summaries: bool = False,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
        stream: bool = False,
        max_iterations: int = 50,
    ) -> Union[str, Dict, Exception, AsyncIterator[str]]:
        """Create and execute a task synchronously."""
        # Use queue to get result from thread
        result_queue = queue.Queue()

        def run_in_thread():
            try:
                # Create a completely new event loop in this thread
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                # Prevent the loop from being closed automatically
                # This keeps resources around until they're done
                thread_loop._close = thread_loop.close
                thread_loop.close = lambda: None

                # Run the async task
                thread_result = thread_loop.run_until_complete(
                    cls.create_async(
                        agent=agent,
                        role=role,
                        goal=goal,
                        attributes=attributes,
                        context=context,
                        instruction=instruction,
                        llm=llm,
                        tools=tools,
                        image_data=image_data,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        require_json_output=require_json_output,
                        callback=callback,
                        event_queue=event_queue,
                        messages=messages,
                        initial_response=initial_response,
                        tool_summaries=tool_summaries,
                        pre_execute=pre_execute,
                        stream=stream,
                        max_iterations=max_iterations,
                    )
                )

                # Put result in queue before any cleanup attempts
                result_queue.put(thread_result)
            except Exception as e:
                logger.error(f"[Task.create] Thread error: {str(e)}", exc_info=True)
                result_queue.put(e)

        # Create and start thread
        thread = threading.Thread(target=run_in_thread)
        thread.start()

        # Wait for thread to complete and get result
        thread.join()
        return result_queue.get()

    @classmethod
    async def create_async(
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
        callback: Optional[
            Callable[[Dict[str, Any]], Union[None, Coroutine[Any, Any, None]]]
        ] = None,
        event_queue: Optional[Any] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        initial_response: bool = False,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
        thread_id: Optional[str] = None,
        tool_summaries: bool = False,
        max_iterations: int = 50,
    ) -> Union[str, Dict, Exception, AsyncIterator[str]]:
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
            max_iterations: Maximum number of tool loop iterations

        Returns:
            Union[str, Exception, AsyncIterator[str]]: Task result or stream

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
                        f"{' Your attributes are: ' + (attributes or (agent.attributes if agent else '')) if attributes or (agent.attributes if agent else '') else ''}."
                        f"{' Additional context: ' + (context or '') if context else ''}"
                    ).strip(),
                }
                messages.insert(0, system_message)

            # Only append if different from last message
            if not messages or messages[-1].get("content") != instruction:
                messages.append({"role": "user", "content": instruction})

            task_data = {
                "agent_id": agent.agent_id if agent else None,
                "role": role or (agent.role if agent else None),
                "goal": goal or (agent.goal if agent else None),
                "attributes": attributes or (agent.attributes if agent else None),
                "context": context,
                "instruction": instruction,
                "llm": llm or (agent.llm if agent else None),
                "tools": set().union(
                    tools or set(), getattr(agent, "tools", None) or set()
                ),  # Merge tools
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
                "thread_id": thread_id,
                "max_iterations": max_iterations,
            }

            # Validate task data using Pydantic
            task = cls.model_validate(task_data)

            logger.debug(f"Created task for agent {task.agent_id or 'unknown'}")
            return await task.execute(callback, pre_execute)

        except Exception as e:
            error_msg = f"Failed to create task: {str(e)}"
            logger.error(error_msg)
            if callback:
                # Check if callback is awaitable before awaiting
                if asyncio.iscoroutinefunction(callback):
                    await callback({"type": "error", "content": error_msg, "thread_id": thread_id})
                else:
                    callback({"type": "error", "content": error_msg, "thread_id": thread_id})
            raise

    async def execute(
        self,
        callback: Optional[
            Callable[[Dict[str, Any]], Union[Any, Coroutine[Any, Any, None]]]
        ] = None,
        pre_execute: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Union[str, Dict, Exception, AsyncIterator[str]]:
        """
        Execute the task with optional tool usage.

        Args:
            callback: Optional progress callback function
            pre_execute: Optional pre-execution callback

        Returns:
            Union[str, Dict, Exception, AsyncIterator[str]]: Task execution result

        Raises:
            Exception: If task execution fails
        """
        try:
            if pre_execute:
                # Check if pre_execute is awaitable before awaiting
                if asyncio.iscoroutinefunction(pre_execute):
                    await pre_execute({"agent_id": self.agent_id})
                else:
                    pre_execute({"agent_id": self.agent_id})

            logger.debug(f"Executing task for agent {self.agent_id or 'unknown'}")

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
                # Check if callback is awaitable before awaiting
                if asyncio.iscoroutinefunction(callback):
                    await callback(
                        {"type": "error", "content": error_msg, "agent_id": self.agent_id}
                    )
                else:
                    callback({"type": "error", "content": error_msg, "agent_id": self.agent_id})
            raise

    async def _direct_llm_call(
        self,
        callback: Optional[
            Callable[[Dict[str, Any]], Union[Any, Coroutine[Any, Any, None]]]
        ] = None,
        response_type: str = "final_response",
    ) -> Union[str, AsyncIterator[str]]:
        """Execute a direct LLM call without tool usage, with fallback support.

        Args:
            callback: Optional callback function for progress updates
            response_type: Type of response event to emit ("final_response" or "initial_response")

        Returns:
            Union[str, AsyncIterator[str]]: Task result or stream

        Raises:
            Exception: If LLM call fails after all fallback attempts
        """
        logger = logging.getLogger("mainframe-orchestra")

        llm_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "require_json_output": self.require_json_output,
            "stream": self.stream,
        }

        # Only add image_data if it exists
        if self.image_data:
            llm_params["image_data"] = self.image_data

        # Convert single LLM to list for unified handling
        llms = [self.llm] if callable(self.llm) else list(self.llm)
        last_error = None

        for i, llm in enumerate(llms, 1):
            try:
                if self.stream:

                    async def stream_wrapper():
                        async for chunk in await llm(messages=self.messages, **llm_params):
                            if callback:
                                # Check if callback is awaitable before awaiting
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(
                                        {
                                            "type": response_type,  # Use passed response_type
                                            "content": chunk,
                                            "agent_id": self.agent_id,
                                            "timestamp": datetime.now().isoformat(),
                                            "streaming": True,
                                        }
                                    )
                                else:
                                    callback(
                                        {
                                            "type": response_type,
                                            "content": chunk,
                                            "agent_id": self.agent_id,
                                            "timestamp": datetime.now().isoformat(),
                                            "streaming": True,
                                        }
                                    )
                            yield chunk

                    return stream_wrapper()

                # Non-streaming response
                if callback and len(llms) > 1:
                    # Check if callback is awaitable before awaiting
                    if asyncio.iscoroutinefunction(callback):
                        await callback(
                            {
                                "type": "fallback_attempt",
                                "content": f"Attempting LLM {i}/{len(llms)}",
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    else:
                        callback(
                            {
                                "type": "fallback_attempt",
                                "content": f"Attempting LLM {i}/{len(llms)}",
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                llm_result = await llm(messages=self.messages, **llm_params)

                # Handle tuple responses (reasoning, error checking)
                if isinstance(llm_result, tuple) and len(llm_result) == 2:
                    response, error = llm_result
                    if error:
                        raise error

                    # Check if response itself is a reasoning tuple
                    if isinstance(response, tuple) and len(response) == 2:
                        reasoning, answer = response
                        # Safely call the callback
                        if callback:
                            callback_event = {
                                "type": "reasoning",
                                "content": reasoning,
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                            if asyncio.iscoroutinefunction(callback):
                                await callback(callback_event)
                            else:
                                callback(callback_event)
                        response = answer  # Use only the answer portion

                elif isinstance(llm_result, dict):
                    response = json.dumps(llm_result)
                elif isinstance(llm_result, str):
                    response = llm_result
                else:
                    raise ValueError(f"Unexpected result type from LLM: {type(llm_result)}")

                if callback:
                    # Check if callback is awaitable before awaiting
                    if asyncio.iscoroutinefunction(callback):
                        await callback(
                            {
                                "type": response_type,  # Use passed response_type
                                "content": response,
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                                "attempt": i if len(llms) > 1 else None,
                            }
                        )
                    else:
                        callback(
                            {
                                "type": response_type,
                                "content": response,
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                                "attempt": i if len(llms) > 1 else None,
                            }
                        )
                # Ensure response is string before strip()
                if not isinstance(response, str):
                    logger.warning(
                        f"Expected string response from LLM, got {type(response)}. Converting to string."
                    )
                    response = str(response)
                return response.strip()

            except Exception as e:
                last_error = e
                logger.error(f"LLM attempt {i}/{len(llms)} failed: {str(e)}")
                if callback:
                    # Check if callback is awaitable before awaiting
                    if asyncio.iscoroutinefunction(callback):
                        await callback(
                            {
                                "type": "error",
                                "content": f"LLM attempt {i}/{len(llms)} failed: {str(e)}",
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    else:
                        callback(
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

        # Ensure all paths return something compatible with the hint
        # If the loop finishes without returning/raising (should not happen with current logic)
        # raise an error consistent with the signature
        raise Exception("LLM call loop completed without returning or raising an error.")

    async def _execute_tool_loop(
        self,
        callback: Optional[
            Callable[[Dict[str, Any]], Union[Any, Coroutine[Any, Any, None]]]
        ] = None,
        pre_execute: Optional[
            Callable[[Dict[str, Any]], Union[None, Coroutine[Any, Any, None]]]
        ] = None,
    ) -> Tuple[Optional[Union[str, Exception]], List[str]]:
        """Execute the tool loop with enhanced logging."""
        logger = logging.getLogger("mainframe-orchestra")
        logger.debug("Starting tool loop execution")

        # Handle case where no tools are provided
        if not self.tools:
            logger.warning("Attempted to execute tool loop, but no tools were provided.")
            return None, []

        try:
            MAX_ITERATIONS = self.max_iterations  # Use the new field
            MAX_IDENTICAL_CALLS = 10
            MAX_CONDUCT_CALLS = 5

            iteration_count = 0
            tool_call_history = {}
            tool_results = []
            conduct_tool_count = 0  # Counter for consecutive conduct tool calls

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
                logger.debug(f"Starting iteration {iteration_count + 1}/{MAX_ITERATIONS}")
                iteration_count += 1

                # Call pre_execute at the beginning of each tool loop iteration
                if pre_execute:
                    # Check if pre_execute is awaitable before awaiting
                    if asyncio.iscoroutinefunction(pre_execute):
                        await pre_execute({"agent_id": self.agent_id})
                    else:
                        pre_execute({"agent_id": self.agent_id})

                # Include tool results in context if we have any
                context_parts = []
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

                # Handle self.llm being a list/tuple
                active_llm = (
                    self.llm[0] if isinstance(self.llm, (list, tuple)) and self.llm else self.llm
                )
                if not callable(active_llm):
                    raise TypeError(f"Resolved LLM is not callable: {type(active_llm)}")

                response, error = await active_llm(
                    messages=temp_history, require_json_output=True, temperature=self.temperature
                )

                if error:
                    logger.error(f"Error from LLM: {error}")
                    if callback:
                        # Check if callback is awaitable before awaiting
                        if asyncio.iscoroutinefunction(callback):
                            await callback({"type": "error", "content": str(error)})
                        else:
                            callback({"type": "error", "content": str(error)})
                    return error, []

                try:
                    # If we got a reasoning tuple, handle both parts
                    if isinstance(response, tuple):
                        reasoning, response = response
                        logger.debug("Received reasoning from LLM")
                        logger.debug(f"Reasoning content: {reasoning}")

                        # Safely call the callback
                        if callback:
                            callback_event = {
                                "type": "reasoning",
                                "content": reasoning,
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                            if asyncio.iscoroutinefunction(callback):
                                await callback(callback_event)
                            else:
                                callback(callback_event)

                        # Log the answer portion
                        logger.debug(f"Answer content: {response}")

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
                        logger.info("Received tool-use completion signal. Tool-use loop exited.")
                        # Safely call the callback
                        if callback:
                            callback_event = {
                                "type": "end_tool_use",
                                "content": "Tool usage complete",
                                "agent_id": self.agent_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                            if asyncio.iscoroutinefunction(callback):
                                await callback(callback_event)
                            else:
                                callback(callback_event)
                        return None, tool_results

                    # Validate each tool call before proceeding
                    tools_dict = {func.__name__: func for func in self.tools}
                    for tool_call in response_data["tool_calls"]:
                        if not isinstance(tool_call, dict):
                            raise ValueError("Each tool call must be an object")

                        if "tool" not in tool_call:
                            raise ValueError("Each tool call must specify a 'tool' name")

                        # Make params default to empty dict if not provided
                        if "params" not in tool_call:
                            tool_call["params"] = {}
                        elif not isinstance(tool_call["params"], dict):
                            raise ValueError("Tool 'params' must be an object")

                        tool_name = tool_call.get("tool")
                        if tool_name not in tools_dict and tool_name != "conduct_tool":
                            raise ValueError(
                                f"Unknown tool: {tool_name}. Available tools: {', '.join(tools_dict.keys())}"
                            )

                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Invalid tool response: {str(e)}"
                    logger.error(f"[TOOL_LOOP] {error_msg}")
                    truncated_response = repr(response[:400])
                    logger.error(f"Problematic response: {truncated_response}...")

                    # Increment retry counter for invalid tool responses
                    retry_count = getattr(self, "_invalid_response_retries", 0) + 1
                    self._invalid_response_retries = retry_count

                    # Exit loop after 3 consecutive failures
                    if retry_count >= 3:
                        logger.warning(
                            f"[TOOL_LOOP] Too many invalid tool responses ({retry_count}). Exiting tool loop."
                        )
                        return None, tool_results

                    # Safely call the callback
                    if callback:
                        callback_event = {
                            "type": "error",
                            "content": error_msg,
                            "response": response[:1000],  # Truncate very long responses
                            "iteration": iteration_count,
                            "retry": retry_count,
                        }
                        if asyncio.iscoroutinefunction(callback):
                            await callback(callback_event)
                        else:
                            callback(callback_event)

                    # Add error to tool results for context in next iteration
                    tool_results.append(
                        "\nTool Response Error:\n"
                        + f"Iteration: {iteration_count}"
                        + "\n"
                        + f"Error: {error_msg}"
                        + "\n"
                        + f"Response: {response[:200]}..."  # Truncate long responses
                    )

                    # Continue to next iteration
                    continue

                # If we get here, all tool calls are valid - proceed with execution
                if "tool_calls" in response_data:
                    # Check initial_response flag BEFORE executing any tools
                    if self.initial_response and iteration_count <= 1:
                        logger.info("Preparing initial response before executing tools")

                        initial_prompt = (
                            f"Given the task instruction: '{self.instruction}' and the planned tool calls: {json.dumps(response_data['tool_calls'], indent=2)}, "
                            "please provide an initial response explaining your planned approach. Provide only a conversational response. You cannot call tools *in* this response, you will be calling them after this response, so don't attempt to call them here. Do not attempt to fully answer the task, and don't ask questions because this will be immediately followed by tool calls."
                        )

                        original_messages = self.messages
                        original_json_requirement = self.require_json_output
                        self.messages = self.messages.copy()
                        self.messages.append({"role": "user", "content": initial_prompt})
                        self.require_json_output = False
                        try:
                            if self.stream:

                                async def callback_wrapper(event):
                                    # Safely call the original callback
                                    if callback:
                                        wrapped_event = {
                                            **event,
                                            "type": "initial_response",
                                            "streaming": True,
                                        }
                                        if asyncio.iscoroutinefunction(callback):
                                            await callback(wrapped_event)
                                        else:
                                            callback(wrapped_event)

                                logger.debug("Starting stream output")
                                # Assuming _direct_llm_call properly handles its own callback logic
                                await self._direct_llm_call(
                                    callback=callback_wrapper, response_type="initial_response"
                                )
                                logger.debug("Stream completed")
                            else:
                                # Assuming _direct_llm_call properly handles its own callback logic
                                await self._direct_llm_call(
                                    callback=callback, response_type="initial_response"
                                )
                        finally:
                            self.require_json_output = original_json_requirement
                            self.messages = original_messages

                    # Now proceed with tool execution
                    for tool_call in response_data["tool_calls"]:
                        if isinstance(tool_call, dict):
                            tool_call_hash = hash_tool_call(tool_call)
                            call_count = tool_call_history.get(tool_call_hash, 0) + 1
                            tool_call_history[tool_call_hash] = call_count

                            if call_count > MAX_IDENTICAL_CALLS:
                                warning_msg = (
                                    f"Exiting tool loop due to verbatim repetition (suggesting infinite loop). "
                                    f"Tool '{tool_call.get('tool')}' with parameters {tool_call.get('params')} "
                                    f"has been called {call_count} times. Maximum allowed repetitions is {MAX_IDENTICAL_CALLS}."
                                )
                                logger.warning(f"[TOOL_LOOP] {warning_msg}")
                                # Safely call the callback
                                if callback:
                                    callback_event = {"type": "warning", "content": warning_msg}
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(callback_event)
                                    else:
                                        callback(callback_event)
                                # Instead of returning an error, return None to proceed to final task
                                return None, tool_results

                            if "task_id" in tool_call:
                                tool_name = "conduct_tool"
                                tool_params = {"instruction": [tool_call]}
                            else:
                                tool_name = tool_call.get("tool")
                                tool_params = tool_call.get("params", {})

                            # Log the tool summary (if available and enabled)
                            if self.tool_summaries and "summary" in tool_call:
                                logger.info(
                                    f"Tool summary for [{tool_name}]: {tool_call['summary']}"
                                )

                            # Send tool call event if a callback is provided
                            if callback:
                                callback_data = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "params": tool_params,
                                    "agent_id": self.agent_id,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                # Add summary to callback_data if available and tool summaries are enabled
                                if self.tool_summaries and "summary" in tool_call:
                                    callback_data["summary"] = tool_call["summary"]
                                # Safely call the callback
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(callback_data)
                                else:
                                    callback(callback_data)

                            # Execute tool and store result
                            # tools_dict is built from non-None self.tools
                            tool_name = tool_call.get("tool")  # Get tool name again
                            # Check tool_name exists before dict access
                            if tool_name and tool_name in tools_dict:
                                tool_func = tools_dict[tool_name]
                            elif tool_name == "conduct_tool":
                                tool_func = None  # Special handling for conduct_tool below
                            else:
                                error_msg = f"Unknown or missing tool name: {tool_name}"
                                logger.error(f"{error_msg}")
                                # Safely call the callback for the error
                                if callback:
                                    callback_event = {"type": "error", "content": error_msg}
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(callback_event)
                                    else:
                                        callback(callback_event)
                                # Store error and continue loop
                                tool_results.append("\nTool Response Error: " + error_msg)
                                continue

                            try:
                                # Handle conduct_tool separately if needed or integrate tool_func check
                                if tool_name == "conduct_tool":
                                    # Assuming conduct_tool logic is handled via special_params
                                    # If conduct_tool needs a callable func, assign it here or adjust logic
                                    pass  # Placeholder for specific conduct_tool logic if needed
                                elif not callable(tool_func):
                                    # This case should be rare if tools_dict is built correctly
                                    raise TypeError(
                                        f"Tool '{tool_name}' retrieved from dict is not callable"
                                    )

                                # Create a copy of tool_params without callback-related items
                                serializable_params = tool_params.copy()
                                special_params = {}

                                if tool_name == "conduct_tool":
                                    # Store callback-related parameters separately
                                    special_params.update(
                                        {
                                            "callback": callback,
                                            "thread_id": self.thread_id,
                                            "event_queue": self.event_queue,
                                            "pre_execute": pre_execute,
                                            "parent_context": self.context,  # Pass the parent task's context
                                        }
                                    )

                                # Log only the serializable parameters
                                logger.info(
                                    f"Executing tool: [{tool_name}] with parameters: {json.dumps(serializable_params, separators=(',', ':'))}"
                                )

                                raw_result = None  # Initialize raw_result
                                # Only attempt to call if tool_func is callable
                                if callable(tool_func):
                                    if asyncio.iscoroutinefunction(tool_func):
                                        # Combine the parameters only for execution
                                        execution_params = {**serializable_params, **special_params}
                                        raw_result = await tool_func(**execution_params)
                                    else:
                                        execution_params = {**serializable_params, **special_params}
                                        raw_result = tool_func(**execution_params)
                                elif tool_name == "conduct_tool":
                                    # Handle conduct_tool case if specific logic is needed here
                                    # For now, we just skip the call as tool_func is None
                                    logger.debug(
                                        f"Skipping call for conduct_tool, parameters prepared in special_params: {special_params.keys()}"
                                    )
                                    # Potentially, conduct_tool logic should happen *here* using special_params
                                    # Since no logic is present, raw_result remains None or could be set to a status message

                                # Check if the result is an exception (raw_result could be None here)
                                if isinstance(raw_result, Exception):
                                    error_msg = f"Tool returned error: {str(raw_result)}"
                                    formatted_error = (
                                        "\nTool Execution Result:\n"
                                        + f"Tool: '{tool_name}'"
                                        + "\n"
                                        + f"Parameters: {json.dumps(tool_params, indent=2)}"
                                        + "\n"
                                        + f"Error: {str(raw_result)}"
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

                                # Safely call the callback with the result
                                if callback:
                                    callback_event = {
                                        "type": "tool_result",
                                        "tool": tool_name,
                                        "result": result_str,
                                        "agent_id": self.agent_id,
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(callback_event)
                                    else:
                                        callback(callback_event)

                                formatted_result = (
                                    "\nTool Execution:\n"
                                    + f"Tool: '{tool_name}'"
                                    + "\n"
                                    + f"Parameters: {json.dumps(tool_params, indent=2)}"
                                    + "\n"
                                    + "Result:"
                                    + "\n"
                                    + result_str
                                )
                                tool_results.append(
                                    formatted_result
                                )  # Using existing tool_results list

                                # After tool execution
                                logger.debug(
                                    f"Result from '{tool_name}': {json.dumps(result, separators=(',', ':'))}"
                                )

                            except Exception as e:
                                error_msg = f"Tool execution error for {tool_name}: {str(e)}"
                                logger.error(f"[TOOL_LOOP] {error_msg}")
                                # Safely call the callback with the error
                                if callback:
                                    callback_event = {"type": "error", "content": error_msg}
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(callback_event)
                                    else:
                                        callback(callback_event)

                                # Format the error as a tool result
                                formatted_error = (
                                    "\nTool Execution Error:\n"
                                    + f"Tool: '{tool_name}'"
                                    + "\n"
                                    + f"Parameters: {json.dumps(tool_params, indent=2)}"
                                    + "\n"
                                    + f"Error: {str(e)}"
                                )
                                tool_results.append(formatted_error)

                                # Continue to the next iteration instead of returning
                                continue

                    # Check if this iteration only contains conduct_tool calls
                    all_conduct_tools = all(
                        tool_call.get("tool") == "conduct_tool"
                        for tool_call in response_data["tool_calls"]
                    )

                    if all_conduct_tools:
                        conduct_tool_count += 1
                        if conduct_tool_count >= MAX_CONDUCT_CALLS:
                            error_msg = f"Maximum consecutive conduct tool calls ({MAX_CONDUCT_CALLS}) reached"
                            logger.warning(f"[TOOL_LOOP] {error_msg}")
                            # Safely call the callback
                            if callback:
                                callback_event = {"type": "error", "content": error_msg}
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(callback_event)
                                else:
                                    callback(callback_event)
                            return None, tool_results  # Return None to allow final response
                    else:
                        # Reset counter if we see other types of tool calls
                        conduct_tool_count = 0

                else:
                    logger.info("[TOOL_LOOP] No tool calls found in response")
                    return None, tool_results

            logger.info(f"Maximum iterations ({MAX_ITERATIONS}) reached")
            # Check for max iterations reached
            if iteration_count >= MAX_ITERATIONS:
                error_msg = f"Maximum tool loop iterations ({MAX_ITERATIONS}) reached"
                logger.error(f"[TOOL_LOOP] {error_msg}")
                # Safely call the callback
                if callback:
                    callback_event = {"type": "error", "content": error_msg}
                    if asyncio.iscoroutinefunction(callback):
                        await callback(callback_event)
                    else:
                        callback(callback_event)
                return Exception(error_msg), tool_results
            # Add a default return path for the function if loop finishes unexpectedly
            return None, []

        except Exception as e:
            error_msg = f"Error in tool loop: {str(e)}"
            logger.error(f"[TOOL_LOOP] {error_msg}")
            # Safely call the callback
            if callback:
                callback_event = {"type": "error", "content": error_msg}
                if asyncio.iscoroutinefunction(callback):
                    await callback(callback_event)
                else:
                    callback(callback_event)
            return e, []

    async def _execute_final_task(
        self,
        tool_results: List[str],
        callback: Optional[
            Callable[[Dict[str, Any]], Union[None, Coroutine[Any, Any, None]]]
        ] = None,
    ) -> Union[str, Dict, Exception, AsyncIterator[str]]:
        """Execute the final task with tool results."""
        logger = logging.getLogger("mainframe-orchestra")

        content_parts = []

        if tool_results:
            content_parts.extend(
                [
                    "\nPrevious Tool Usage:",
                    "".join(tool_results),
                    "\nYou have just completed and exited your tool-use phase, and you are now writing your final response. Do not make any more tool calls.",
                ]
            )

        content_parts.append("Now focus on addressing the instruction:\n" + self.instruction)

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
                    return parse_json_response(result)
                except ValueError as e:
                    return ValueError(
                        f"Failed to parse JSON from LLM response: {result}" + "\nError: " + str(e)
                    )

            return result

        except Exception as e:
            logger.error(f"[FINAL_TASK] Error in final task execution: {str(e)}")
            if callback:
                # Check if callback is awaitable before awaiting
                if asyncio.iscoroutinefunction(callback):
                    await callback({"type": "error", "content": str(e)})
                else:
                    callback({"type": "error", "content": str(e)})
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
            end (str): String to append after each chunk when using the default print callback
            flush (bool): Whether to flush after each chunk when using the default print callback

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
                # Check if chunk is dict before accessing keys
                if isinstance(chunk, dict):
                    # Handle streaming responses (both initial and final)
                    if chunk.get("type") in ["initial_response", "final_response"] and chunk.get(
                        "streaming"
                    ):
                        content = chunk.get("content", "")  # Use .get() for safer access
                        collected.append(str(content))
                        if callback:
                            if callback is print:  # Use 'is' to check for specific print function
                                print(content, end=end, flush=flush)
                            else:
                                # Call other callbacks without end/flush
                                callback(str(content))
                    # Handle non-streaming responses
                    elif chunk.get("type") in ["initial_response", "final_response"]:
                        content = chunk.get("content", "")  # Use .get()
                        chunk_type = chunk.get("type", "response")  # Use .get()
                        collected.append(str(content))
                        if callback:
                            if callback is print:
                                print(
                                    "\n" + chunk_type.replace("_", " ").title() + f": {content}",
                                    end=end,  # OK for print
                                    flush=flush,  # OK for print
                                )
                            else:
                                # Call other callbacks without end/flush
                                callback(str(content))
                # Handle direct string chunks
                elif isinstance(chunk, str):
                    collected.append(chunk)
                    if callback:
                        if callback is print:
                            print(chunk, end=end, flush=flush)
                        else:
                            # Call other callbacks without end/flush
                            callback(chunk)

            # Add newline after stream is complete (only for print)
            if callback is print:
                # Ensure callback is callable before calling (Fixing L1012)
                if callable(callback):
                    print("\n", end="", flush=True)

            return "".join(collected)

        # Ensure inner function has a return path
        return loop.run_until_complete(process())
