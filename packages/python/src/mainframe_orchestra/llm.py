# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import json
import logging
import os
import random
import re
import time
import requests
import base64
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union, Callable, ClassVar, Any, cast

import google.generativeai as genai
import ollama
from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    APIResponseValidationError as AnthropicResponseValidationError,
)
from anthropic import (
    APIStatusError as AnthropicStatusError,
)
from anthropic import (
    APITimeoutError as AnthropicTimeoutError,
)
from anthropic import (
    AsyncAnthropic,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic._types import NOT_GIVEN
from anthropic.types import TextBlock
from google.ai.generativelanguage_v1beta.types import Content, Part
from halo import Halo
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from openai import (
    APIConnectionError as OpenAIConnectionError,
)
from openai import (
    APIError as OpenAIAPIError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    AsyncOpenAI,
)
from openai import (
    AuthenticationError as OpenAIAuthenticationError,
)
from openai import (
    BadRequestError as OpenAIBadRequestError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.chat import ChatCompletionChunk

from .utils.braintrust_utils import wrap_openai
from .utils.parse_json_response import parse_json_response

# Import the configured logger
from .utils.logging_config import logger

# Import config, fall back to environment variables if not found
try:
    from .config import config
except ImportError:
    import os

    class EnvConfig:
        def __init__(self):
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            self.TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
            self.HF_TOKEN = os.getenv("HF_TOKEN")

        def validate_api_key(self, key_name: str) -> str:
            """Retrieves API key and raises error if missing."""
            api_key = getattr(self, key_name, None)
            if not api_key:
                raise ValueError(f"{key_name} not found in environment variables.")
            return api_key

    config = EnvConfig()

# Schema for the expected tool call JSON response
TOOL_CALLS_SCHEMA = {
    "name": "execute_tool_calls",
    "description": "Determine the next tool calls to execute or indicate that tool use is complete.",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "description": "A list of tool calls to request. Leave empty if no further tool calls are required.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "The name of the tool to be called."
                        },
                        "params": {
                            "type": "object",
                            "description": "The parameters required for the tool call.",
                            "additionalProperties": True
                        },
                        "summary": {
                            "type": ["string", "null"],
                            "description": "Optional summary explaining the reason for the tool call."
                        }
                    },
                    "required": ["tool", "params", "summary"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["tool_calls"],
        "additionalProperties": False
    }
}

# Global settings
verbosity = False
debug = False

# Retry settings
MAX_RETRIES = 3
BASE_DELAY = 1
MAX_DELAY = 10


def set_verbosity(value: Union[str, bool, int]):
    global verbosity, debug
    if isinstance(value, str):
        value = value.lower()
        if value in ["debug", "2"]:
            verbosity = True
            debug = True
            logger.setLevel(logging.DEBUG)
        elif value in ["true", "1"]:
            verbosity = True
            debug = False
            logger.setLevel(logging.INFO)
        else:
            verbosity = False
            debug = False
            logger.setLevel(logging.WARNING)
    elif isinstance(value, bool):
        verbosity = value
        debug = False
        logger.setLevel(logging.INFO if value else logging.WARNING)
    elif isinstance(value, int):
        if value == 2:
            verbosity = True
            debug = True
            logger.setLevel(logging.DEBUG)
        elif value == 1:
            verbosity = True
            debug = False
            logger.setLevel(logging.INFO)
        else:
            verbosity = False
            debug = False
            logger.setLevel(logging.WARNING)


class OpenAICompatibleProvider:
    """
    Base class for handling OpenAI-compatible API providers.
    This handles providers that use the OpenAI API format but with different base URLs.
    """

    @staticmethod
    async def _prepare_image_data(
        image_data: Union[str, List[str]], provider_name: str
    ) -> Union[str, List[str]]:
        """Prepare image data according to provider requirements"""
        if not image_data:
            return image_data

        images = [image_data] if isinstance(image_data, str) else image_data
        processed_images = []

        for img in images:
            if img.startswith(("http://", "https://")):
                # Download and convert URL to base64
                response = requests.get(img)
                response.raise_for_status()
                base64_data = base64.b64encode(response.content).decode("utf-8")

                if provider_name in ["OpenAI", "Gemini"]:
                    # These providers need data URL format
                    processed_images.append(f"data:image/jpeg;base64,{base64_data}")
                else:
                    # Others can handle raw base64
                    processed_images.append(base64_data)
            else:
                # Handle existing base64 data
                if provider_name in ["OpenAI", "Gemini"] and not img.startswith("data:"):
                    # Add data URL prefix if missing
                    processed_images.append(f"data:image/jpeg;base64,{img}")
                else:
                    processed_images.append(img)

        return processed_images[0] if isinstance(image_data, str) else processed_images

    @staticmethod
    async def send_request(
        model: str,
        provider_name: str,
        base_url: Optional[str],
        api_key: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """Sends a request to an OpenAI-compatible API provider"""
        try:
            # Process image data if present
            if image_data:
                image_data = await OpenAICompatibleProvider._prepare_image_data(
                    image_data, provider_name
                )

            spinner = Halo(text=f"Sending request to {provider_name}...", spinner="dots")
            spinner.start()

            # Initialize client
            if base_url:
                client = wrap_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
            else:
                client = wrap_openai(AsyncOpenAI(api_key=api_key))

            # Separate system message (instructions) from input messages
            system_instructions = None
            input_messages = []
            if messages:
                for msg in messages:
                    if msg["role"] == "system":
                        system_instructions = msg["content"]
                    else:
                        input_messages.append(msg)
            else:
                input_messages = []

            # Prepare top-level request parameters
            request_params = {
                "model": model,
                "input": input_messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens, # Renamed from max_tokens
            }

            # Add instructions if a system message was found
            if system_instructions:
                request_params["instructions"] = system_instructions

            # Prepare the 'text' parameter ONLY if JSON output is required
            request_text_params = None
            if require_json_output:
                request_text_params = {
                    "format": {
                        "type": "json_schema",
                        "name": TOOL_CALLS_SCHEMA["name"],
                        "schema": TOOL_CALLS_SCHEMA["schema"],
                        "strict": TOOL_CALLS_SCHEMA.get("strict", True)
                    }
                }
                request_params["text"] = request_text_params

            # Log request details (adjusted for new structure)
            logger.debug(
                f"[LLM] {provider_name} ({model}) Request (Responses API): {json.dumps(request_params | {'stream': stream}, separators=(',', ':'))}"
            )

            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started (Responses API)")
                    try:
                        # Pass stream=True directly in the main call
                        response_stream = await client.responses.create(
                            **request_params,
                            stream=True
                        )
                        async for chunk in response_stream:
                            if isinstance(chunk, ResponseTextDeltaEvent):
                                if chunk.delta:
                                    content = chunk.delta
                                    full_message += content
                                    yield content

                        logger.debug("Stream complete (Responses API)")
                        logger.debug(f"Full message: {full_message}")
                        yield "\n" # Add newline after stream completion
                    except OpenAIAuthenticationError as e:
                        logger.error(
                            f"Authentication failed: Please check your {provider_name} API key. Error: {str(e)}"
                        )
                        yield "" # Return empty on error
                    except OpenAIBadRequestError as e:
                        logger.error(f"Invalid request parameters: {str(e)}")
                        yield ""
                    except (OpenAIConnectionError, OpenAITimeoutError) as e:
                        logger.error(f"Connection error: {str(e)}")
                        yield ""
                    except OpenAIRateLimitError as e:
                        logger.error(f"Rate limit exceeded: {str(e)}")
                        yield ""
                    except OpenAIAPIError as e:
                        logger.error(f"{provider_name} API error: {str(e)}")
                        yield ""
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during streaming: {e}")
                        yield ""

                return stream_generator()

            # Non-streaming logic for responses.create
            spinner.text = f"Waiting for {model} response (Responses API)..."
            response = await client.responses.create(
                **request_params,
                stream=False
            )

            # Access response content (assuming output_text)
            content = response.output_text if hasattr(response, 'output_text') else ""
            spinner.succeed("Request completed (Responses API)")

            # Process JSON responses
            if require_json_output and content:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(',', ':'))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    logger.error(f"Failed to parse response as JSON: {content}. Error: {e}")
                    return "", e # Return error if JSON parsing fails

            # For non-JSON responses
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except OpenAIAuthenticationError as e:
            spinner.fail("Authentication failed")
            logger.error(
                f"Authentication failed: Please check your {provider_name} API key. Error: {str(e)}"
            )
            return "", e
        except OpenAIBadRequestError as e:
            spinner.fail("Invalid request")
            logger.error(f"Invalid request parameters: {str(e)}")
            return "", e
        except (OpenAIConnectionError, OpenAITimeoutError) as e:
            spinner.fail("Connection failed")
            logger.error(f"Connection error: {str(e)}")
            return "", e
        except OpenAIRateLimitError as e:
            spinner.fail("Rate limit exceeded")
            logger.error(f"Rate limit exceeded: {str(e)}")
            return "", e
        except OpenAIAPIError as e:
            spinner.fail("API Error")
            logger.error(f"{provider_name} API error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return "", e
        finally:
            if spinner.spinner_id:  # Check if spinner is still running
                spinner.stop()


class OpenAIChatCompletionsProvider:
    """
    Provider class for handling OpenAI-compatible APIs using the chat.completions endpoint.
    """

    @staticmethod
    async def send_request(
        model: str,
        provider_name: str,
        base_url: Optional[str],
        api_key: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """Sends a request using client.chat.completions.create"""
        try:
            # Use the image preparation logic from the other provider
            if image_data:
                image_data = await OpenAICompatibleProvider._prepare_image_data(
                    image_data, provider_name
                )

            spinner = Halo(text=f"Sending request to {provider_name} (Chat)...", spinner="dots")
            spinner.start()

            # Initialize client
            if base_url:
                client = wrap_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
            else:
                client = wrap_openai(AsyncOpenAI(api_key=api_key))

            # --- Start: Adapt messages for chat.completions.create ---
            chat_messages = []
            if messages:
                system_prompt = None
                for i, msg in enumerate(messages):
                    # Find and store the first system message
                    if msg["role"] == "system" and system_prompt is None:
                        if isinstance(msg.get("content"), str):
                            system_prompt = msg["content"]
                            chat_messages.append({"role": "system", "content": system_prompt})
                        else:
                            logger.warning(f"System prompt content is not a string: {type(msg.get('content'))}. Ignoring.")
                    # Add user and assistant messages directly
                    elif msg["role"] in ["user", "assistant"]:
                        # Ensure content structure is compatible (str or list of blocks)
                        current_content = msg.get("content")
                        if isinstance(current_content, (str, list)):
                            chat_messages.append({"role": msg["role"], "content": current_content})
                        else:
                            logger.warning(f"Unsupported content type {type(current_content)} in role {msg['role']}. Converting to string.")
                            chat_messages.append({"role": msg["role"], "content": str(current_content)})

                    # Ignore other system messages or roles
                    elif msg["role"] == "system":
                         logger.warning("Multiple system messages found. Only the first one is used in chat.completions.")
                    else:
                         logger.warning(f"Unsupported role '{msg['role']}' ignored for chat.completions.")

            # Append image data to the last user message if necessary
            if image_data:
                last_user_msg_index = -1
                for i in range(len(chat_messages) - 1, -1, -1):
                    if chat_messages[i]["role"] == "user":
                        last_user_msg_index = i
                        break

                if last_user_msg_index != -1:
                    user_content = chat_messages[last_user_msg_index]["content"]
                    # Ensure content is a list to append image data
                    if isinstance(user_content, str):
                        chat_messages[last_user_msg_index]["content"] = [{"type": "text", "text": user_content}]
                    elif not isinstance(user_content, list):
                        logger.error("Cannot add image data: Last user message content is not string or list.")
                        # Optionally raise an error or handle differently
                        raise ValueError("Invalid message format for adding images")

                    # Append image(s) - assuming _prepare_image_data provides ready-to-use URLs
                    images_to_append = [image_data] if isinstance(image_data, str) else image_data
                    for img_url_data in images_to_append: # Now expecting data URLs
                         # Ensure img_url_data is a string before appending
                         if isinstance(img_url_data, str):
                            chat_messages[last_user_msg_index]["content"].append({"type": "image_url", "image_url": {"url": img_url_data}})
                         else:
                             logger.warning(f"Skipping non-string image data: {type(img_url_data)}")

                else:
                    logger.warning("Image data provided but no user message found to attach it to.")
            # --- End: Adapt messages --- #

            # --- Start: Prepare request parameters for chat.completions.create ---
            request_params = {
                "model": model,
                "messages": chat_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if require_json_output:
                request_params["response_format"] = { "type": "json_object" }
                logger.info(f"Requesting JSON object output format for {model}.")
            # --- End: Prepare request parameters --- #

            log_params = request_params.copy()
            log_params.pop('stream', None)
            logger.debug(
                f"[LLM] {provider_name} ({model}) Request (Chat Completions API): {json.dumps(log_params | {'stream': stream}, separators=(',', ':'))}"
            )

            if stream:
                spinner.stop()
                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started (Chat Completions API)")
                    try:
                        response_stream = await client.chat.completions.create(**request_params)
                        async for chunk in response_stream:
                            if isinstance(chunk, ChatCompletionChunk) and chunk.choices:
                                delta = chunk.choices[0].delta
                                if delta and delta.content:
                                    content = delta.content
                                    full_message += content
                                    yield content
                        logger.debug("Stream complete (Chat Completions API)")
                        logger.debug(f"Full message: {full_message}")
                    except OpenAIAuthenticationError as e:
                        logger.error(f"Authentication failed: {str(e)}")
                        yield f"Error: Authentication Failed ({provider_name})"
                    except OpenAIBadRequestError as e:
                        logger.error(f"Invalid request parameters: {str(e)}")
                        yield f"Error: Invalid Request ({str(e)})"
                    except (OpenAIConnectionError, OpenAITimeoutError) as e:
                        logger.error(f"Connection error: {str(e)}")
                        yield f"Error: Connection Error ({str(e)})"
                    except OpenAIRateLimitError as e:
                        logger.error(f"Rate limit exceeded: {str(e)}")
                        yield f"Error: Rate Limit Exceeded ({str(e)})"
                    except OpenAIAPIError as e:
                        logger.error(f"{provider_name} API error: {str(e)}")
                        yield f"Error: API Error ({provider_name} - {str(e)})"
                    except Exception as e:
                        logger.error(f"Unexpected streaming error: {e}", exc_info=True)
                        yield f"Error: Unexpected Streaming Error ({str(e)})"
                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response (Chat Completions API)..."
            response = await client.chat.completions.create(**request_params)
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            spinner.succeed(f"Request completed ({provider_name} - Chat Completions API)")

            if require_json_output and content:
                try:
                    # Validate it's JSON, but return the raw string
                    parse_json_response(content)
                    logger.debug(f"[LLM] API Response (JSON Validated): {content[:100]}...") # Log truncated
                    return content, None
                except ValueError as e:
                    logger.error(f"LLM returned non-JSON despite json_object request: {content}. Error: {e}")
                    return content, ValueError("LLM returned non-JSON despite json_object format request")

            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except OpenAIAuthenticationError as e:
            spinner.fail("Authentication failed")
            logger.error(f"Authentication failed: {str(e)}")
            return "", e
        except OpenAIBadRequestError as e:
            spinner.fail("Invalid request")
            logger.error(f"Invalid request parameters: {str(e)}")
            return "", e
        except (OpenAIConnectionError, OpenAITimeoutError) as e:
            spinner.fail("Connection failed")
            logger.error(f"Connection error: {str(e)}")
            return "", e
        except OpenAIRateLimitError as e:
            spinner.fail("Rate limit exceeded")
            logger.error(f"Rate limit exceeded: {str(e)}")
            return "", e
        except OpenAIAPIError as e:
            spinner.fail("API Error")
            logger.error(f"{provider_name} API error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return "", e
        finally:
            if spinner.spinner_id:
                spinner.stop()


class OpenaiModels:
    """
    Class containing methods for interacting with OpenAI models.
    """

    # Class variable to store a default base URL for all requests
    _default_base_url = None

    @classmethod
    def set_base_url(cls, base_url: str) -> None:
        """
        Set a default base URL for all OpenAI requests.
        """
        cls._default_base_url = base_url
        logger.info(f"Set default OpenAI base URL to: {base_url}")

    @classmethod
    async def send_openai_request(
        cls,
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
        base_url: Optional[str] = None,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to an OpenAI model asynchronously and handles retries.
        """
        # Process images if present
        if image_data and messages:
            last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user_msg:
                content = []
                if isinstance(image_data, str):
                    image_data = [image_data]

                for image in image_data:
                    if image.startswith(("http://", "https://")):
                        content.append({"type": "image_url", "image_url": {"url": image}})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            }
                        )

                # Add original text content
                content.append({"type": "text", "text": last_user_msg["content"]})
                last_user_msg["content"] = content

        # Add check for non-streaming models (currently only o1 models) at the start
        if stream and model in ["o1-mini", "o1-preview"]:
            logger.error(
                f"Streaming is not supported for {model}. Falling back to non-streaming request."
            )
            stream = False

        # Get API key
        api_key = config.validate_api_key("OPENAI_API_KEY")

        # Use provided base_url, or fall back to class default, or config/env
        custom_base_url = (
            base_url or cls._default_base_url or getattr(config, "OPENAI_BASE_URL", None)
        )

        # Handle o series model temperature
        if model.startswith("o"):
            temperature = 1.0

        return await OpenAICompatibleProvider.send_request(
            model=model,
            provider_name="OpenAI",
            base_url=custom_base_url,
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            base_url: Optional[str] = None,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await OpenaiModels.send_openai_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
                base_url=base_url,
            )

        return wrapper

    # Model-specific methods using custom_model
    gpt_4_turbo: ClassVar[Callable] = custom_model("gpt-4-turbo")
    gpt_3_5_turbo: ClassVar[Callable] = custom_model("gpt-3.5-turbo")
    gpt_4: ClassVar[Callable] = custom_model("gpt-4")
    gpt_4o: ClassVar[Callable] = custom_model("gpt-4o")
    gpt_4o_mini: ClassVar[Callable] = custom_model("gpt-4o-mini")
    gpt_4_1: ClassVar[Callable] = custom_model("gpt-4.1")
    gpt_4_1_mini: ClassVar[Callable] = custom_model("gpt-4.1-mini")
    gpt_4_1_nano: ClassVar[Callable] = custom_model("gpt-4.1-nano")
    gpt_4_5_preview: ClassVar[Callable] = custom_model("gpt-4.5-preview")
    o1_mini: ClassVar[Callable] = custom_model("o1-mini")
    o1_preview: ClassVar[Callable] = custom_model("o1-preview")
    o1: ClassVar[Callable] = custom_model("o1")
    o3_mini: ClassVar[Callable] = custom_model("o3-mini")
    o3: ClassVar[Callable] = custom_model("o3")
    o4_mini: ClassVar[Callable] = custom_model("o4-mini")


class AnthropicModels:
    """
    Class containing methods for interacting with Anthropic models using the Messages API.
    """

    @staticmethod
    async def send_anthropic_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        max_retries: int = 3,  # Add max_retries parameter
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends an asynchronous request to an Anthropic model using the Messages API format.
        Implements automatic retries for rate limit errors with backoff.
        """
        spinner = Halo(text="Sending request to Anthropic...", spinner="dots")
        spinner.start()
        backoff_times = [10, 30, 60]  # Backoff in seconds

        try:
            api_key = config.validate_api_key("ANTHROPIC_API_KEY")
            client = AsyncAnthropic(api_key=api_key)
            if not client.api_key:
                raise ValueError("Anthropic API key not found in environment variables.")

            # Convert OpenAI format messages to Anthropic Messages API format
            anthropic_messages = []
            system_message = None

            # Process provided messages or create from prompts
            if messages:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    # Handle system messages separately
                    if role == "system":
                        # Ensure system content is a string
                        if isinstance(content, str):
                            system_message = content
                        else:
                            logger.warning("System message content must be a string. Ignoring non-string system message.")
                            system_message = None # Or keep previous if applicable
                    elif role == "user":
                        # Ensure user content is correctly formatted for Anthropic (list or str)
                        anthropic_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        # Ensure assistant content is correctly formatted (str)
                        if isinstance(content, str):
                            anthropic_messages.append({"role": "assistant", "content": content})
                        else:
                             logger.warning("Assistant message content must be a string. Skipping non-string assistant message.")
                    elif role == "function":
                        # Convert function result to string for user message
                        func_result_str = str(content)
                        anthropic_messages.append(
                            {"role": "user", "content": f"Function result: {func_result_str}"}
                        )

            # If JSON output is required, add instruction to the system message
            if require_json_output:
                json_instruction = "Do not comment before or after the JSON, or provide backticks or language declarations, return only the JSON object."

                # If we have a system message, append the instruction
                if system_message is not None:
                    if isinstance(system_message, str):
                        system_message += f"\n\n{json_instruction}"
                    else:
                        # This case should ideally not happen if system messages are always strings
                        logger.warning("System message content is not a string, cannot append JSON instruction.")
                else:
                    # If no system message exists, create one
                    system_message = json_instruction

            # Handle image data if present
            if image_data:
                if isinstance(image_data, str):
                    image_data = [image_data]

                # Add images to the last user message or create new one
                last_msg = (
                    anthropic_messages[-1]
                    if anthropic_messages
                    else {"role": "user", "content": []}
                )
                if last_msg["role"] != "user":
                    last_msg = {"role": "user", "content": []}
                    anthropic_messages.append(last_msg)

                # Convert content to list if it's a string
                if isinstance(last_msg["content"], str):
                    last_msg["content"] = [{"type": "text", "text": last_msg["content"]}]
                elif not isinstance(last_msg["content"], list):
                    last_msg["content"] = []

                # Add each image
                for img in image_data:
                    if img.startswith(("http://", "https://")):
                        # For URLs, we need to download and convert to base64
                        try:
                            response = requests.get(img)
                            response.raise_for_status()
                            image_base64 = base64.b64encode(response.content).decode("utf-8")
                            last_msg["content"].append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_base64,
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"Failed to process image URL: {str(e)}")
                            raise
                    else:
                        # For base64 data, use it directly
                        last_msg["content"].append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img,
                                },
                            }
                        )

            # Log request details
            logger.debug(
                f"[LLM] Anthropic ({model}) Request: {json.dumps({'system_message': system_message, 'messages': anthropic_messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stop_sequences': stop_sequences}, separators=(',', ':'))}"
            )

            # Handle streaming
            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")

                    for attempt in range(max_retries):
                        try:
                            response = await client.messages.create(
                                model=model,
                                messages=anthropic_messages,
                                system=system_message if isinstance(system_message, str) else NOT_GIVEN,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stop_sequences=stop_sequences if stop_sequences else NOT_GIVEN,
                                stream=True,
                            )
                            async for chunk in response:
                                if chunk.type == "content_block_delta":
                                    if chunk.delta.type == "text_delta":
                                        content = chunk.delta.text
                                        full_message += content
                                        yield content
                                elif chunk.type == "message_delta":
                                    # When a stop_reason is provided, log it without per-chunk verbosity
                                    if chunk.delta.stop_reason:
                                        logger.debug(
                                            f"Message delta stop reason: {chunk.delta.stop_reason}"
                                        )
                                elif chunk.type == "error":
                                    logger.error(f"Stream error: {chunk.error}")
                                    break
                            logger.debug("Stream complete")
                            logger.debug(f"Final message: {full_message}")
                            break  # Exit retry loop on success

                        except AnthropicRateLimitError as e:
                            if attempt < max_retries - 1:  # If not the last attempt
                                backoff_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                                logger.warning(f"Rate limit exceeded during streaming, backing off for {backoff_time} seconds (Attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(backoff_time)
                                continue
                            logger.error(f"Rate limit exceeded during streaming after {max_retries} attempts: {str(e)}", exc_info=True)
                            yield ""

                        except (AnthropicConnectionError, AnthropicTimeoutError) as e:
                            logger.error(f"Connection error during streaming: {str(e)}", exc_info=True)
                            yield ""
                            break
                        except AnthropicStatusError as e:
                            logger.error(f"API status error during streaming: {str(e)}", exc_info=True)
                            yield ""
                            break
                        except AnthropicResponseValidationError as e:
                            logger.error(
                                f"Invalid response format during streaming: {str(e)}", exc_info=True
                            )
                            yield ""
                            break
                        except ValueError as e:
                            logger.error(
                                f"Configuration error during streaming: {str(e)}", exc_info=True
                            )
                            yield ""
                            break
                        except Exception as e:
                            logger.error(
                                f"An unexpected error occurred during streaming: {e}", exc_info=True
                            )
                            yield ""
                            break

                return stream_generator()

            # Non-streaming logic with retries
            for attempt in range(max_retries):
                try:
                    spinner.text = f"Waiting for {model} response..."
                    response = await client.messages.create(
                        model=model,
                        messages=anthropic_messages,
                        system=system_message if isinstance(system_message, str) else NOT_GIVEN,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop_sequences=stop_sequences if stop_sequences else NOT_GIVEN,
                    )

                    # Extract content, handling potential ToolUseBlock
                    content = ""
                    if response.content:
                        first_block = response.content[0]
                        if isinstance(first_block, TextBlock):
                            content = first_block.text
                        else:
                            # Handle ToolUseBlock or other block types if needed
                            logger.debug(f"First content block is not text: {type(first_block)}")

                    spinner.succeed("Request completed")
                    # For non-JSON responses, keep original formatting but make single line
                    logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
                    return content.strip(), None

                except AnthropicRateLimitError as e:
                    if attempt < max_retries - 1:  # If not the last attempt
                        backoff_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                        spinner.text = f"Rate limit hit, retrying in {backoff_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                        logger.warning(f"Rate limit exceeded, backing off for {backoff_time} seconds")
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        spinner.fail("Rate limit exceeded after all retries")
                        logger.error(f"Rate limit exceeded after {max_retries} attempts: {str(e)}", exc_info=True)
                        return "", e

                except (AnthropicConnectionError, AnthropicTimeoutError) as e:
                    spinner.fail("Connection failed")
                    logger.error(f"Connection error: {str(e)}", exc_info=True)
                    return "", e
                except AnthropicStatusError as e:
                    spinner.fail("API Status Error")
                    logger.error(f"API Status Error: {str(e)}", exc_info=True)
                    return "", e
                except AnthropicResponseValidationError as e:
                    spinner.fail("Invalid Response Format")
                    logger.error(f"Invalid response format: {str(e)}", exc_info=True)
                    return "", e
                except ValueError as e:
                    spinner.fail("Configuration Error")
                    logger.error(f"Configuration error: {str(e)}", exc_info=True)
                    return "", e
                except Exception as e:
                    spinner.fail("Request failed")
                    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                    return "", e

        except Exception as e:
            spinner.fail("Setup failed")
            logger.error(f"Failed to set up request: {str(e)}", exc_info=True)
            return "", e
        finally:
            if spinner.spinner_id:  # Check if spinner is still running
                spinner.stop()

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stop_sequences: Optional[List[str]] = None,
            stream: bool = False,  # Add stream parameter
        ) -> Union[
            Tuple[str, Optional[Exception]], AsyncGenerator[str, None]
        ]:  # Update return type
            return await AnthropicModels.send_anthropic_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stop_sequences=stop_sequences,
                stream=stream,  # Pass stream parameter
            )

        return wrapper

    # Model-specific methods using custom_model
    opus: ClassVar[Callable] = custom_model("claude-3-opus-latest")
    sonnet: ClassVar[Callable] = custom_model("claude-3-sonnet-20240229")
    haiku: ClassVar[Callable] = custom_model("claude-3-haiku-20240307")
    sonnet_3_5: ClassVar[Callable] = custom_model("claude-3-5-sonnet-latest")
    haiku_3_5: ClassVar[Callable] = custom_model("claude-3-5-haiku-latest")
    sonnet_3_7: ClassVar[Callable] = custom_model("claude-3-7-sonnet-latest")


class OpenrouterModels:
    """
    Class containing methods for interacting with OpenRouter models.
    """

    @classmethod
    async def send_openrouter_request(
        cls,
        model: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to OpenRouter models.
        """
        # Process images if present
        if image_data and messages:
            last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user_msg:
                content = []
                if isinstance(image_data, str):
                    image_data = [image_data]

                for image in image_data:
                    if image.startswith(("http://", "https://")):
                        content.append({"type": "image_url", "image_url": {"url": image}})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            }
                        )

                # Add original text content
                content.append({"type": "text", "text": last_user_msg["content"]})
                last_user_msg["content"] = content

        # Get API key
        api_key = config.validate_api_key("OPENROUTER_API_KEY")

        return await OpenAIChatCompletionsProvider.send_request(
            model=model,
            provider_name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await OpenrouterModels.send_openrouter_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
            )

        return wrapper

    # Model-specific methods using custom_model
    haiku: ClassVar[Callable] = custom_model("anthropic/claude-3-haiku")
    haiku_3_5: ClassVar[Callable] = custom_model("anthropic/claude-3.5-haiku")
    sonnet: ClassVar[Callable] = custom_model("anthropic/claude-3-sonnet")
    sonnet_3_5: ClassVar[Callable] = custom_model("anthropic/claude-3.5-sonnet")
    sonnet_3_7: ClassVar[Callable] = custom_model("anthropic/claude-3.7-sonnet")
    opus: ClassVar[Callable] = custom_model("anthropic/claude-3-opus")
    gpt_3_5_turbo: ClassVar[Callable] = custom_model("openai/gpt-3.5-turbo")
    gpt_4_turbo: ClassVar[Callable] = custom_model("openai/gpt-4-turbo")
    gpt_4: ClassVar[Callable] = custom_model("openai/gpt-4")
    gpt_4o: ClassVar[Callable] = custom_model("openai/gpt-4o")
    gpt_4o_mini: ClassVar[Callable] = custom_model("openai/gpt-4o-mini")
    gpt_4_5_preview: ClassVar[Callable] = custom_model("openai/gpt-4.5-preview")
    o1_preview: ClassVar[Callable] = custom_model("openai/o1-preview")
    o1_mini: ClassVar[Callable] = custom_model("openai/o1-mini")
    o3: ClassVar[Callable] = custom_model("openai/o3")
    o3_mini: ClassVar[Callable] = custom_model("openai/o3-mini")
    o3_mini_high: ClassVar[Callable] = custom_model("openai/o3-mini-high")
    o4_mini: ClassVar[Callable] = custom_model("openai/o4-mini")
    o4_mini_high: ClassVar[Callable] = custom_model("openai/o4-mini-high")
    gpt_4_1: ClassVar[Callable] = custom_model("openai/gpt-4.1")
    gpt_4_1_mini: ClassVar[Callable] = custom_model("openai/gpt-4.1-mini")
    gpt_4_1_nano: ClassVar[Callable] = custom_model("openai/gpt-4.1-nano")
    gemini_flash_1_5: ClassVar[Callable] = custom_model("google/gemini-flash-1.5")
    llama_3_70b_sonar_32k: ClassVar[Callable] = custom_model("perplexity/llama-3-sonar-large-32k-chat")
    command_r: ClassVar[Callable] = custom_model("cohere/command-r-plus")
    nous_hermes_2_mistral_7b_dpo: ClassVar[Callable] = custom_model("nousresearch/nous-hermes-2-mistral-7b-dpo")
    nous_hermes_2_mixtral_8x7b_dpo: ClassVar[Callable] = custom_model("nousresearch/nous-hermes-2-mixtral-8x7b-dpo")
    nous_hermes_yi_34b: ClassVar[Callable] = custom_model("nousresearch/nous-hermes-yi-34b")
    qwen_2_72b: ClassVar[Callable] = custom_model("qwen/qwen-2-72b-instruct")
    mistral_7b: ClassVar[Callable] = custom_model("mistralai/mistral-7b-instruct")
    mistral_7b_nitro: ClassVar[Callable] = custom_model("mistralai/mistral-7b-instruct:nitro")
    mixtral_8x7b_instruct: ClassVar[Callable] = custom_model("mistralai/mixtral-8x7b-instruct")
    mixtral_8x7b_instruct_nitro: ClassVar[Callable] = custom_model("mistralai/mixtral-8x7b-instruct:nitro")
    mixtral_8x22b_instruct: ClassVar[Callable] = custom_model("mistralai/mixtral-8x22b-instruct")
    wizardlm_2_8x22b: ClassVar[Callable] = custom_model("microsoft/wizardlm-2-8x22b")
    neural_chat_7b: ClassVar[Callable] = custom_model("intel/neural-chat-7b")
    gemma_7b_it: ClassVar[Callable] = custom_model("google/gemma-7b-it")
    gemini_pro: ClassVar[Callable] = custom_model("google/gemini-pro")
    llama_3_8b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3-8b-instruct")
    llama_3_70b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3-70b-instruct")
    llama_3_70b_instruct_nitro: ClassVar[Callable] = custom_model("meta-llama/llama-3-70b-instruct:nitro")
    llama_3_8b_instruct_nitro: ClassVar[Callable] = custom_model("meta-llama/llama-3-8b-instruct:nitro")
    dbrx_132b_instruct: ClassVar[Callable] = custom_model("databricks/dbrx-instruct")
    deepseek_coder: ClassVar[Callable] = custom_model("deepseek/deepseek-coder")
    llama_3_1_70b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3.1-70b-instruct")
    llama_3_1_8b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3.1-8b-instruct")
    llama_3_1_405b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3.1-405b-instruct")
    qwen_2_5_coder_32b_instruct: ClassVar[Callable] = custom_model("qwen/qwen-2.5-coder-32b-instruct")
    claude_3_5_haiku: ClassVar[Callable] = custom_model("anthropic/claude-3-5-haiku")
    ministral_8b: ClassVar[Callable] = custom_model("mistralai/ministral-8b")
    ministral_3b: ClassVar[Callable] = custom_model("mistralai/ministral-3b")
    llama_3_1_nemotron_70b_instruct: ClassVar[Callable] = custom_model("nvidia/llama-3.1-nemotron-70b-instruct")
    gemini_flash_1_5_8b: ClassVar[Callable] = custom_model("google/gemini-flash-1.5-8b")
    llama_3_2_3b_instruct: ClassVar[Callable] = custom_model("meta-llama/llama-3.2-3b-instruct")


class OllamaModels:
    @staticmethod
    async def call_ollama(
        model: str,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        stream: bool = False,  # Add stream parameter
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:  # Update return type
        """
        Updated to handle messages array format compatible with Task class.
        """
        logger.debug(
            f"Parameters: model={model}, messages={messages}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}"
        )

        spinner = Halo(text="Sending request to Ollama...", spinner="dots")
        spinner.start()

        try:
            # Process messages into Ollama format
            if not messages:
                messages = []

            # Handle image data by appending to messages
            if image_data:
                logger.debug("Processing image data")
                if isinstance(image_data, str):
                    image_data = [image_data]

                # Add images to the last user message or create new one
                last_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
                if last_msg:
                    # Append images to existing user message
                    current_content = last_msg["content"]
                    # Ensure content is a string before appending images for Ollama
                    if isinstance(current_content, str):
                        for i, image in enumerate(image_data, start=1):
                            current_content += f"\n<image>{image}</image>"
                        last_msg["content"] = current_content
                    else:
                        logger.warning("Ollama image handling expects existing text content. Found non-string content, skipping image append.")
                else:
                    # Create new message with images
                    image_content = "\n".join(f"<image>{img}</image>" for img in image_data)
                    messages.append({"role": "user", "content": image_content})

            logger.debug(f"Final messages structure: {messages}")

            for attempt in range(MAX_RETRIES):
                logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
                try:
                    client = ollama.Client()

                    logger.debug(
                        f"[LLM] Ollama ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}"
                    )

                    if stream:
                        spinner.stop()  # Stop spinner before streaming

                        async def stream_generator():
                            full_message = ""
                            logger.debug("Stream started")
                            try:
                                response = client.chat(
                                    model=model,
                                    messages=messages,
                                    format="json" if require_json_output else None,
                                    options={"temperature": temperature, "num_predict": max_tokens},
                                    stream=True,
                                )

                                for chunk in response:
                                    if (
                                        chunk
                                        and "message" in chunk
                                        and "content" in chunk["message"]
                                    ):
                                        content = chunk["message"]["content"]
                                        full_message += content
                                        yield content
                                logger.debug("Stream completed")
                                logger.debug(f"Final streamed message: {full_message}")
                            except Exception as e:
                                logger.error(f"Streaming error: {str(e)}")
                                yield ""

                        return stream_generator()

                    # Non-streaming logic
                    response = client.chat(
                        model=model,
                        messages=messages,
                        format="json" if require_json_output else None,
                        options={"temperature": temperature, "num_predict": max_tokens},
                    )

                    response_text = response["message"]["content"]

                    # verbosity printing before json parsing
                    logger.info(f"[LLM] API Response: {response_text.strip()}")

                    if require_json_output:
                        try:
                            json_response = parse_json_response(response_text)
                        except ValueError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                        return json.dumps(json_response), None

                    # For non-JSON responses, keep original formatting but make single line
                    logger.debug(
                        f"[LLM] API Response: {' '.join(response_text.strip().splitlines())}"
                    )
                    return response_text.strip(), None

                except ollama.ResponseError as e:
                    logger.error(f"Ollama response error: {e}")
                    logger.debug(f"ResponseError details: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2**attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        logger.info(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except ollama.RequestError as e:
                    logger.error(f"Ollama request error: {e}")

                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2**attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        logger.info(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except Exception as e:
                    logger.error(f"An unexpected error occurred: {e}")
                    logger.debug(f"Unexpected error details: {type(e).__name__}, {e}")
                    return "", e

        finally:
            if spinner.spinner_id:  # Check if spinner is still running
                spinner.stop()

        return "", Exception("Max retries reached")

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            stream: bool = False,  # Add stream parameter
        ) -> Union[
            Tuple[str, Optional[Exception]], AsyncGenerator[str, None]
        ]:  # Update return type
            return await OllamaModels.call_ollama(
                model=model_name,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,  # Pass stream parameter
            )

        return wrapper


class TogetheraiModels:
    """
    Class containing methods for interacting with Together AI models.
    """

    @classmethod
    async def send_together_request(
        cls,
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to Together AI using the messages API format.
        """
        # Get API key
        api_key = config.validate_api_key("TOGETHERAI_API_KEY")

        # Process images if present
        if image_data and messages:
            last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user_msg:
                content = []
                if isinstance(image_data, str):
                    image_data = [image_data]

                for i, image in enumerate(image_data, start=1):
                    content.append({"type": "text", "text": f"Image {i}:"})
                    if image.startswith(("http://", "https://")):
                        content.append({"type": "image_url", "image_url": {"url": image}})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            }
                        )

                # Add original text content
                content.append({"type": "text", "text": last_user_msg["content"]})
                last_user_msg["content"] = content

        return await OpenAIChatCompletionsProvider.send_request(
            model=model,
            provider_name="Together AI",
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await TogetheraiModels.send_together_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
            )

        return wrapper

    # Model-specific methods using custom_model
    meta_llama_3_1_70b_instruct_turbo: ClassVar[Callable] = custom_model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")


class GroqModels:
    """
    Class containing methods for interacting with Groq models.
    """

    @classmethod
    async def send_groq_request(
        cls,
        model: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to Groq models.
        """
        # Get API key
        api_key = config.validate_api_key("GROQ_API_KEY")

        return await OpenAIChatCompletionsProvider.send_request(
            model=model,
            provider_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await GroqModels.send_groq_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
            )

        return wrapper

    # Model-specific methods using custom_model
    gemma2_9b_it: ClassVar[Callable] = custom_model("gemma2-9b-it")
    llama_3_3_70b_versatile: ClassVar[Callable] = custom_model("llama-3.3-70b-versatile")
    llama_3_1_8b_instant: ClassVar[Callable] = custom_model("llama-3.1-8b-instant")
    llama_guard_3_8b: ClassVar[Callable] = custom_model("llama-guard-3-8b")
    llama3_70b_8192: ClassVar[Callable] = custom_model("llama3-70b-8192")
    llama3_8b_8192: ClassVar[Callable] = custom_model("llama3-8b-8192")
    mixtral_8x7b_32768: ClassVar[Callable] = custom_model("mixtral-8x7b-32768")
    llama_3_2_vision: ClassVar[Callable] = custom_model("llama-3.2-11b-vision-preview")


class GeminiModels:
    """
    Class containing methods for interacting with Google's Gemini models using the chat format.
    """

    @staticmethod
    async def send_gemini_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Union[str, Tuple[str, Optional[Exception]]], None]:
        """
        Sends a request to Gemini using the chat format.
        Yields: string chunks if stream=True, OR a single Tuple(result, error) if stream=False.
        """
        spinner = Halo(text=f"Sending request to Gemini ({model})...", spinner="dots")

        try:
            # Start spinner
            spinner.start()

            # Process image data if present
            if image_data:
                if isinstance(image_data, str):
                    image_data = [image_data]

                processed_images = []
                for img in image_data:
                    if img.startswith(("http://", "https://")):
                        # Download image from URL
                        response = requests.get(img)
                        response.raise_for_status()
                        img_data = response.content
                        processed_images.append({"mime_type": "image/jpeg", "data": img_data})
                    else:
                        # Assume it's base64, decode it
                        if img.startswith("data:"):
                            # Strip data URL prefix if present
                            img = img.split(",", 1)[1]
                        img_bytes = base64.b64decode(img)
                        processed_images.append({"mime_type": "image/jpeg", "data": img_bytes})

                image_data = processed_images

            # Configure API and model
            api_key = config.validate_api_key("GEMINI_API_KEY")
            genai.configure(api_key=api_key)

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if require_json_output:
                generation_config.update({"response_mime_type": "application/json"})

            model_instance = genai.GenerativeModel(
                model_name=model, generation_config=genai.GenerationConfig(**generation_config)
            )
            # Print all messages together after spinner starts
            logger.debug(
                f"[LLM] Gemini ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}"
            )

            if stream:
                spinner.stop()
                # Handle None case for messages before reversing
                last_user_message_content = ""
                if messages is not None:
                    last_user_message_content = next(
                        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
                        "" # Default if no user message found
                    )
                # Ensure content is string for Gemini stream start
                if not isinstance(last_user_message_content, str):
                    logger.warning("Cannot start Gemini stream with non-string content. Using empty string.")
                    last_user_message_content = ""

                full_message = ""
                logger.debug("Stream started")

                try:
                    response = model_instance.generate_content(last_user_message_content, stream=True)
                    for chunk in response:
                        if chunk.text:
                            content = chunk.text
                            full_message += content
                            yield content # Yields str

                    logger.debug("Stream complete")
                    logger.debug(f"Full message: {full_message}")
                except Exception as e:
                    logger.error(f"Gemini streaming error: {str(e)}")
                    yield "", e # Yields Tuple[str, Exception]

            else:
                # Non-streaming: Use chat format
                chat = model_instance.start_chat(history=[])

                # Process messages and images
                if messages:
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]

                        if role == "user":
                            if image_data and msg == messages[-1]:
                                parts = []
                                for img in image_data:
                                    parts.append(img)  # Now using processed image data
                                parts.append(content)
                                response = chat.send_message(parts)
                            else:
                                response = chat.send_message(content)
                        elif role == "assistant":
                            # Ensure content is a string before adding to Gemini history
                            if isinstance(content, str):
                                part = Part(text=content)
                                history_entry = Content(role="model", parts=[part])
                                chat.history.append(history_entry)
                            else:
                                logger.warning("Cannot add non-string assistant content to Gemini chat history.")

                text_output = ""
                if 'response' in locals() and hasattr(response, 'text'):
                    text_output = response.text.strip()
                else:
                    logger.warning("Gemini response object not found or lacks text attribute.")

                spinner.succeed("Request completed")

                # Print response
                logger.debug(f"[LLM] API Response: {text_output.strip()}")

                if require_json_output:
                    try:
                        parsed = json.loads(text_output)
                        yield json.dumps(parsed), None # Yields Tuple[str, None]
                    except ValueError as ve:
                        logger.error(f"Failed to parse Gemini response as JSON: {ve}")
                        yield "", ve # Yields Tuple[str, ValueError]
                else:
                    yield text_output, None # Yields Tuple[str, None]

        except Exception as e:
            spinner.fail("Gemini request failed")
            logger.error(f"Unexpected error for Gemini model ({model}): {str(e)}")
            yield "", e # Yields Tuple[str, Exception]

    @staticmethod
    def custom_model(model_name: str):
        # This wrapper returns EITHER an AsyncGenerator OR a Tuple
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
             # send_gemini_request ALWAYS returns an async generator object
            result_generator = GeminiModels.send_gemini_request(
                 model=model_name,
                 image_data=image_data,
                 temperature=temperature,
                 max_tokens=max_tokens,
                 require_json_output=require_json_output,
                 messages=messages,
                 stream=stream,
            )

            if stream:
                # If streaming is requested, return the generator directly.
                # We need to ensure it yields only 'str' as per the wrapper's return hint.
                async def stream_wrapper() -> AsyncGenerator[str, None]:
                    try:
                        async for item in result_generator:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, tuple): # Handle tuple yielded on error during stream
                                content, error = item
                                logger.error(f"Error yielded during Gemini stream: {error}")
                                yield f"Error during stream: {error}" # Yield error as string
                            else:
                                logger.warning(f"Unexpected type yielded during Gemini stream: {type(item)}")
                                yield f"Unexpected stream item type: {type(item)}"
                    except Exception as e:
                         logger.error(f"Exception iterating Gemini stream wrapper: {e}", exc_info=True)
                         yield f"Error iterating stream: {e}"
                return stream_wrapper()
            else:
                # If not streaming, consume the single Tuple item from the generator
                try:
                    # The non-streaming path yields a single tuple: (content, error)
                    async for item in result_generator:
                         if isinstance(item, tuple) and len(item) == 2:
                             content, error = item
                             if isinstance(content, str) and (error is None or isinstance(error, Exception)):
                                 return item # Return the tuple directly
                             else:
                                 logger.error(f"Unexpected item structure in tuple from non-streaming Gemini generator: ({type(content)}, {type(error)})")
                                 return "", Exception("Unexpected tuple structure from Gemini generator")
                         else:
                             # This case should ideally not happen in non-streaming mode
                             logger.error(f"Unexpected item type yielded from non-streaming Gemini generator: {type(item)}")
                             return "", Exception("Unexpected item type from non-streaming Gemini")

                    # If the generator finishes without yielding (e.g., immediate exception before first yield)
                    logger.error("Non-streaming Gemini generator finished unexpectedly without yielding a result.")
                    return "", Exception("Gemini generator finished unexpectedly")
                except Exception as e:
                     logger.error(f"Error consuming non-streaming Gemini generator: {e}", exc_info=True)
                     return "", e
        return wrapper

    # Model-specific methods using custom_model
    gemini_1_5_flash: ClassVar[Callable] = custom_model("gemini-1.5-flash")
    gemini_1_5_flash_8b: ClassVar[Callable] = custom_model("gemini-1.5-flash-8b")
    gemini_1_5_pro: ClassVar[Callable] = custom_model("gemini-1.5-pro")
    gemini_2_0_flash: ClassVar[Callable] = custom_model("gemini-2.0-flash")
    gemini_2_5_flash: ClassVar[Callable] = custom_model("gemini-2.5-flash-preview-04-17")
    gemini_2_5_pro: ClassVar[Callable] = custom_model("gemini-2.5-pro-exp-03-25")


class DeepseekModels:
    """
    Class containing methods for interacting with DeepSeek models.
    """

    @staticmethod
    def _preprocess_reasoner_messages(
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]],
        require_json_output: bool = False
    ) -> List[Dict[str, str]]:
        """
        Transform messages specifically for the DeepSeek Reasoner model.
        """
        if require_json_output:
            logger.warning(
                "Warning: JSON output format is not supported for the Reasoner model. Request will proceed without JSON formatting."
            )

        if not messages:
            return []

        processed = []
        current_role = None
        current_content = []

        for msg in messages:
            # Ensure content is string before processing
            content_val = msg.get("content")
            if not isinstance(content_val, str):
                 # Handle non-string content (e.g., log, convert, skip)
                 logger.warning(f"Skipping non-string content in message for DeepSeek reasoner: {content_val}")
                 content_str = str(content_val) # Convert to string as fallback
            else:
                 content_str = content_val

            if msg["role"] == current_role:
                # Same role as previous message, append content
                current_content.append(content_str)
            else:
                # Different role, flush previous message if exists
                if current_role:
                    processed.append({"role": current_role, "content": "\n".join(current_content)})
                # Start new message
                current_role = msg["role"]
                current_content = [content_str]

        if current_role:
            processed.append({"role": current_role, "content": "\n".join(current_content)})

        return processed

    @classmethod
    async def send_deepseek_request(
        cls,
        model: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to DeepSeek models.
        """
        # Get API key
        api_key = config.validate_api_key("DEEPSEEK_API_KEY")

        # Apply model-specific transformations
        if model == "deepseek-reasoner":
            processed_messages = cls._preprocess_reasoner_messages(messages, require_json_output)
            # Correct cast syntax and ensure it points to the new provider
            messages = cast(Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]], processed_messages)
            # Disable JSON output for reasoner model
            require_json_output = False

        return await OpenAIChatCompletionsProvider.send_request(
            model=model,
            provider_name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await DeepseekModels.send_deepseek_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
            )

        return wrapper

    # Model-specific methods
    chat: ClassVar[Callable] = custom_model("deepseek-chat")
    reasoner: ClassVar[Callable] = custom_model("deepseek-reasoner")


class HuggingFaceModels:
    """
    Class containing methods for interacting with HuggingFace models via InferenceClient.
    """

    @staticmethod
    async def send_huggingface_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to HuggingFace Inference API asynchronously and handles retries.
        """
        spinner = Halo(text="Sending request to HuggingFace...", spinner="dots")
        spinner.start()

        try:
            # Initialize the client with API token from environment
            client = InferenceClient(model=model, token=config.HF_TOKEN)

            # Convert messages to prompt format
            prompt = ""
            if messages:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "system":
                        prompt += f"<|system|>{content}\n"
                    elif role == "user":
                        prompt += f"<|user|>{content}\n"
                    elif role == "assistant":
                        prompt += f"<|assistant|>{content}\n"

            # Handle image inputs if present
            if image_data:
                if isinstance(image_data, str):
                    image_data = [image_data]
                for img in image_data:
                    if prompt:
                        prompt += "\n"
                    prompt += f"<image>{img}</image>"

            logger.debug(
                f"[LLM] HuggingFace ({model}) Request: {json.dumps({'prompt': prompt, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}"
            )

            if stream:
                spinner.stop()

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")
                    try:
                        response = client.text_generation(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )

                        for chunk in response:
                            # Ensure chunk is not a string before trying attribute access
                            if not isinstance(chunk, str):
                                token = getattr(chunk, 'token', None)
                                if token and hasattr(token, 'text') and token.text:
                                    content = token.text
                                    # Clean the content of unwanted tags for each chunk
                                    content = HuggingFaceModels._clean_response_tags(content)
                                    full_message += content
                                    yield content

                        logger.debug("Stream complete")
                        logger.debug(f"Full message: {full_message}")
                    except Exception as e:
                        logger.error(f"An error occurred during streaming: {e}", exc_info=True)
                        yield ""

                return stream_generator()

            spinner.text = f"Waiting for {model} response..."

            parameters = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            }

            if require_json_output:
                parameters["do_sample"] = False  # More deterministic for JSON outputs

            response = client.text_generation(prompt, **parameters)

            # Clean the response of unwanted tags
            content = HuggingFaceModels._clean_response_tags(response)
            spinner.succeed("Request completed")

            # Handle JSON output if required
            if require_json_output:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(",", ":"))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    return "", e

            # For non-JSON responses, keep original formatting but make single line
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except HfHubHTTPError as e:
            spinner.fail("HuggingFace Token Error")
            if e.response.status_code == 401:
                logger.error("Authentication failed: Please check your HuggingFace token")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded")
            else:
                logger.error(f"HuggingFace API error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}")
            return "", e
        finally:
            if spinner.spinner_id:
                spinner.stop()

    @staticmethod
    def _clean_response_tags(text: str) -> str:
        """
        Clean HuggingFace model responses by removing unwanted tags.

        Args:
            text: The text response from the model

        Returns:
            Cleaned text with unwanted tags removed
        """
        if not text:
            return text

        # Remove common tag patterns that appear in HuggingFace model responses
        # This handles tags like <||, <|assistant|>, etc.
        cleaned = re.sub(r"<\|[^>]*\|>", "", text)

        # Handle incomplete tags at the beginning or end
        cleaned = re.sub(r"^<\|.*?(?=\w)", "", cleaned)  # Beginning of text
        cleaned = re.sub(r"(?<=\w).*?\|>$", "", cleaned)  # End of text

        # Handle other special cases
        cleaned = re.sub(r"<\|\|", "", cleaned)
        cleaned = re.sub(r"<\|", "", cleaned)
        cleaned = re.sub(r"\|>", "", cleaned)

        return cleaned.strip()

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            return await HuggingFaceModels.send_huggingface_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream,
            )

        return wrapper

    # Add commonly used models
    qwen2_5_coder: ClassVar[Callable] = custom_model("Qwen/Qwen2.5-Coder-32B-Instruct")
    meta_llama_3_8b: ClassVar[Callable] = custom_model("meta-llama/Meta-Llama-3-8B-Instruct")
