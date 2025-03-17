# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import json
import logging
import os
import random
import re
import time
import requests
import base64
from typing import AsyncGenerator, Dict, Iterator, List, Optional, Tuple, Union

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

    config = EnvConfig()

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
        base_url: str,
        api_key: str,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        additional_params: Optional[Dict] = None,
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
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
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url

            client = wrap_openai(AsyncOpenAI(**client_kwargs))

            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Add max_tokens if provided
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Add JSON output format if required
            if require_json_output:
                request_params["response_format"] = {"type": "json_object"}

            # Add any additional parameters
            if additional_params:
                request_params.update(additional_params)

            # Log request details
            logger.debug(
                f"[LLM] {provider_name} ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}"
            )

            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")
                    try:
                        stream_params = {**request_params, "stream": True}
                        response = await client.chat.completions.create(**stream_params)
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_message += content
                                yield content
                        logger.debug("Stream complete")
                        logger.debug(f"Full message: {full_message}")
                        yield "\n"
                    except OpenAIAuthenticationError as e:
                        logger.error(
                            f"Authentication failed: Please check your {provider_name} API key. Error: {str(e)}"
                        )
                        yield ""
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

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = await client.chat.completions.create(**request_params)

            content = response.choices[0].message.content
            spinner.succeed("Request completed")

            # Process JSON responses
            if require_json_output:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(",", ":"))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    return "", e

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

    @staticmethod
    def _transform_o1_messages(
        messages: List[Dict[str, str]], require_json_output: bool = False
    ) -> List[Dict[str, str]]:
        """
        Transform messages for o1 models by handling system messages and JSON requirements.
        """
        modified_messages = []
        system_content = ""

        # Extract system message if present
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break

        # Add system content as a user message if present
        if system_content:
            modified_messages.append(
                {"role": "user", "content": f"[System Instructions]\n{system_content}"}
            )

        # Process remaining messages
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                content = msg["content"]
                if require_json_output and msg == messages[-1]:  # If this is the last user message
                    content += "\n\nDo NOT include backticks, language declarations, or commentary before or after the JSON content."
                modified_messages.append({"role": "user", "content": content})
            else:
                modified_messages.append(msg)

        return modified_messages

    @classmethod
    async def send_openai_request(
        cls,
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        base_url: Optional[str] = None,
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
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

        # Handle o1 model message transformations
        additional_params = None
        if model in ["o1-mini", "o1-preview"]:
            messages = cls._transform_o1_messages(messages, require_json_output)
            # o1 models use max_completion_tokens instead of max_tokens
            additional_params = {"max_completion_tokens": max_tokens}
            # Set max_tokens to None as it's not used
            max_tokens = None
            # Override temperature for o1 models - they only support temperature=1
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
            stream=stream,
            additional_params=additional_params,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
            stream: bool = False,
            base_url: Optional[str] = None,
        ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
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
    gpt_4_turbo = custom_model("gpt-4-turbo")
    gpt_3_5_turbo = custom_model("gpt-3.5-turbo")
    gpt_4 = custom_model("gpt-4")
    gpt_4o = custom_model("gpt-4o")
    gpt_4o_mini = custom_model("gpt-4o-mini")
    o1_mini = custom_model("o1-mini")
    o1_preview = custom_model("o1-preview")
    gpt_4_5_preview = custom_model("gpt-4.5-preview")


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
        messages: Optional[List[Dict[str, str]]] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends an asynchronous request to an Anthropic model using the Messages API format.
        """
        spinner = Halo(text="Sending request to Anthropic...", spinner="dots")
        spinner.start()

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
                        system_message = content  # Store the system message from messages
                    elif role == "user":
                        anthropic_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        anthropic_messages.append({"role": "assistant", "content": content})
                    elif role == "function":
                        anthropic_messages.append(
                            {"role": "user", "content": f"Function result: {content}"}
                        )

            # If JSON output is required, add instruction to the system message
            if require_json_output:
                json_instruction = "Do not comment before or after the JSON, or provide backticks or language declarations, return only the JSON object."

                # If we have a system message, append the instruction
                if system_message is not None:
                    system_message += f"\n\n{json_instruction}"
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
                    try:
                        response = await client.messages.create(
                            model=model,
                            messages=anthropic_messages,
                            system=system_message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop_sequences=stop_sequences if stop_sequences else None,
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
                    except (AnthropicConnectionError, AnthropicTimeoutError) as e:
                        logger.error(f"Connection error during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except AnthropicRateLimitError as e:
                        logger.error(
                            f"Rate limit exceeded during streaming: {str(e)}", exc_info=True
                        )
                        yield ""
                    except AnthropicStatusError as e:
                        logger.error(f"API status error during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except AnthropicResponseValidationError as e:
                        logger.error(
                            f"Invalid response format during streaming: {str(e)}", exc_info=True
                        )
                        yield ""
                    except ValueError as e:
                        logger.error(
                            f"Configuration error during streaming: {str(e)}", exc_info=True
                        )
                        yield ""
                    except Exception as e:
                        logger.error(
                            f"An unexpected error occurred during streaming: {e}", exc_info=True
                        )
                        yield ""

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = await client.messages.create(
                model=model,
                messages=anthropic_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences if stop_sequences else None,
            )

            content = response.content[0].text if response.content else ""
            spinner.succeed("Request completed")
            # For non-JSON responses, keep original formatting but make single line
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except (AnthropicConnectionError, AnthropicTimeoutError) as e:
            spinner.fail("Connection failed")
            logger.error(f"Connection error: {str(e)}", exc_info=True)
            return "", e
        except AnthropicRateLimitError as e:
            spinner.fail("Rate limit exceeded")
            logger.error(f"Rate limit exceeded: {str(e)}", exc_info=True)
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
            messages: Optional[List[Dict[str, str]]] = None,
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
    opus = custom_model("claude-3-opus-latest")
    sonnet = custom_model("claude-3-sonnet-20240229")
    haiku = custom_model("claude-3-haiku-20240307")
    sonnet_3_5 = custom_model("claude-3-5-sonnet-latest")
    haiku_3_5 = custom_model("claude-3-5-haiku-latest")
    sonnet_3_7 = custom_model("claude-3-7-sonnet-latest")


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
        messages: Optional[List[Dict[str, str]]] = None,
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

        # Handle o1 model message transformations if needed
        additional_params = None
        if model.endswith("o1-mini") or model.endswith("o1-preview"):
            messages = OpenaiModels._transform_o1_messages(messages, require_json_output)

        return await OpenAICompatibleProvider.send_request(
            model=model,
            provider_name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream,
            additional_params=additional_params,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
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
    haiku = custom_model("anthropic/claude-3-haiku")
    haiku_3_5 = custom_model("anthropic/claude-3.5-haiku")
    sonnet = custom_model("anthropic/claude-3-sonnet")
    sonnet_3_5 = custom_model("anthropic/claude-3.5-sonnet")
    sonnet_3_7 = custom_model("anthropic/claude-3.7-sonnet")
    opus = custom_model("anthropic/claude-3-opus")
    gpt_3_5_turbo = custom_model("openai/gpt-3.5-turbo")
    gpt_4_turbo = custom_model("openai/gpt-4-turbo")
    gpt_4 = custom_model("openai/gpt-4")
    gpt_4o = custom_model("openai/gpt-4o")
    gpt_4o_mini = custom_model("openai/gpt-4o-mini")
    gpt_4_5_preview = custom_model("openai/gpt-4.5-preview")
    o1_preview = custom_model("openai/o1-preview")
    o1_mini = custom_model("openai/o1-mini")
    gemini_flash_1_5 = custom_model("google/gemini-flash-1.5")
    llama_3_70b_sonar_32k = custom_model("perplexity/llama-3-sonar-large-32k-chat")
    command_r = custom_model("cohere/command-r-plus")
    nous_hermes_2_mistral_7b_dpo = custom_model("nousresearch/nous-hermes-2-mistral-7b-dpo")
    nous_hermes_2_mixtral_8x7b_dpo = custom_model("nousresearch/nous-hermes-2-mixtral-8x7b-dpo")
    nous_hermes_yi_34b = custom_model("nousresearch/nous-hermes-yi-34b")
    qwen_2_72b = custom_model("qwen/qwen-2-72b-instruct")
    mistral_7b = custom_model("mistralai/mistral-7b-instruct")
    mistral_7b_nitro = custom_model("mistralai/mistral-7b-instruct:nitro")
    mixtral_8x7b_instruct = custom_model("mistralai/mixtral-8x7b-instruct")
    mixtral_8x7b_instruct_nitro = custom_model("mistralai/mixtral-8x7b-instruct:nitro")
    mixtral_8x22b_instruct = custom_model("mistralai/mixtral-8x22b-instruct")
    wizardlm_2_8x22b = custom_model("microsoft/wizardlm-2-8x22b")
    neural_chat_7b = custom_model("intel/neural-chat-7b")
    gemma_7b_it = custom_model("google/gemma-7b-it")
    gemini_pro = custom_model("google/gemini-pro")
    llama_3_8b_instruct = custom_model("meta-llama/llama-3-8b-instruct")
    llama_3_70b_instruct = custom_model("meta-llama/llama-3-70b-instruct")
    llama_3_70b_instruct_nitro = custom_model("meta-llama/llama-3-70b-instruct:nitro")
    llama_3_8b_instruct_nitro = custom_model("meta-llama/llama-3-8b-instruct:nitro")
    dbrx_132b_instruct = custom_model("databricks/dbrx-instruct")
    deepseek_coder = custom_model("deepseek/deepseek-coder")
    llama_3_1_70b_instruct = custom_model("meta-llama/llama-3.1-70b-instruct")
    llama_3_1_8b_instruct = custom_model("meta-llama/llama-3.1-8b-instruct")
    llama_3_1_405b_instruct = custom_model("meta-llama/llama-3.1-405b-instruct")
    qwen_2_5_coder_32b_instruct = custom_model("qwen/qwen-2.5-coder-32b-instruct")
    claude_3_5_haiku = custom_model("anthropic/claude-3-5-haiku")
    ministral_8b = custom_model("mistralai/ministral-8b")
    ministral_3b = custom_model("mistralai/ministral-3b")
    llama_3_1_nemotron_70b_instruct = custom_model("nvidia/llama-3.1-nemotron-70b-instruct")
    gemini_flash_1_5_8b = custom_model("google/gemini-flash-1.5-8b")
    llama_3_2_3b_instruct = custom_model("meta-llama/llama-3.2-3b-instruct")


class OllamaModels:
    @staticmethod
    async def call_ollama(
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
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
                    for i, image in enumerate(image_data, start=1):
                        current_content += f"\n<image>{image}</image>"
                    last_msg["content"] = current_content
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
            messages: Optional[List[Dict[str, str]]] = None,
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
        messages: Optional[List[Dict[str, str]]] = None,
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

        return await OpenAICompatibleProvider.send_request(
            model=model,
            provider_name="Together AI",
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream,
            additional_params=None,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
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
    meta_llama_3_1_70b_instruct_turbo = custom_model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")


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
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
        """
        Sends a request to Groq models.
        """
        # Get API key
        api_key = config.validate_api_key("GROQ_API_KEY")

        return await OpenAICompatibleProvider.send_request(
            model=model,
            provider_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json_output=require_json_output,
            messages=messages,
            stream=stream,
            additional_params=None,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
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
    gemma2_9b_it = custom_model("gemma2-9b-it")
    llama_3_3_70b_versatile = custom_model("llama-3.3-70b-versatile")
    llama_3_1_8b_instant = custom_model("llama-3.1-8b-instant")
    llama_guard_3_8b = custom_model("llama-guard-3-8b")
    llama3_70b_8192 = custom_model("llama3-70b-8192")
    llama3_8b_8192 = custom_model("llama3-8b-8192")
    mixtral_8x7b_32768 = custom_model("mixtral-8x7b-32768")
    llama_3_2_vision = custom_model("llama-3.2-11b-vision-preview")


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
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to Gemini using the chat format.
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
                last_user_message = next(
                    (msg["content"] for msg in reversed(messages) if msg["role"] == "user"), ""
                )
                full_message = ""
                logger.debug("Stream started")

                try:
                    response = model_instance.generate_content(last_user_message, stream=True)
                    for chunk in response:
                        if chunk.text:
                            content = chunk.text
                            full_message += content
                            yield content

                    logger.debug("Stream complete")
                    logger.debug(f"Full message: {full_message}")
                except Exception as e:
                    logger.error(f"Gemini streaming error: {str(e)}")
                    yield ""
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
                            chat.history.append({"role": "model", "parts": [content]})

                # Get the final response
                text_output = response.text.strip()
                spinner.succeed("Request completed")

                # Print response
                logger.debug(f"[LLM] API Response: {text_output.strip()}")

                if require_json_output:
                    try:
                        parsed = json.loads(text_output)
                        yield json.dumps(parsed)
                    except ValueError as ve:
                        logger.error(f"Failed to parse Gemini response as JSON: {ve}")
                        yield ""
                else:
                    yield text_output

        except Exception as e:
            spinner.fail("Gemini request failed")
            logger.error(f"Unexpected error for Gemini model ({model}): {str(e)}")
            yield ""

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
            stream: bool = False,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            if stream:
                # For streaming, simply return the asynchronous generator.
                return GeminiModels.send_gemini_request(
                    model=model_name,
                    image_data=image_data,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    require_json_output=require_json_output,
                    messages=messages,
                    stream=True,
                )
            else:
                # For non-streaming, consume the entire async generator,
                # accumulating all yielded chunks into a single string.
                result = ""
                async for chunk in GeminiModels.send_gemini_request(
                    model=model_name,
                    image_data=image_data,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    require_json_output=require_json_output,
                    messages=messages,
                    stream=False,
                ):
                    result += chunk
                return result, None

        return wrapper

    # Model-specific methods using custom_model
    # gemini_2_0_flash = custom_model("gemini-2.0-flash")  # Experimental
    gemini_1_5_flash = custom_model("gemini-1.5-flash")
    gemini_1_5_flash_8b = custom_model("gemini-1.5-flash-8b")
    gemini_1_5_pro = custom_model("gemini-1.5-pro")


class DeepseekModels:
    """
    Class containing methods for interacting with DeepSeek models.
    """

    @staticmethod
    def _preprocess_reasoner_messages(
        messages: List[Dict[str, str]], require_json_output: bool = False
    ) -> List[Dict[str, str]]:
        """
        Transform messages specifically for the DeepSeek Reasoner model.
        """
        if require_json_output:
            logger.warning(
                "Warning: JSON output format is not supported for the Reasoner model. Request will proceed without JSON formatting."
            )

        if not messages:
            return messages

        processed = []
        current_role = None
        current_content = []

        for msg in messages:
            if msg["role"] == current_role:
                # Same role as previous message, append content
                current_content.append(msg["content"])
            else:
                # Different role, flush previous message if exists
                if current_role:
                    processed.append({"role": current_role, "content": "\n".join(current_content)})
                # Start new message
                current_role = msg["role"]
                current_content = [msg["content"]]

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
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to DeepSeek models.
        """
        # Get API key
        api_key = config.validate_api_key("DEEPSEEK_API_KEY")

        # Apply model-specific transformations
        additional_params = None
        if model == "deepseek-reasoner":
            messages = cls._preprocess_reasoner_messages(messages, require_json_output)
            # Disable JSON output for reasoner model
            require_json_output = False

        return await OpenAICompatibleProvider.send_request(
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
            additional_params=additional_params,
        )

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
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
    chat = custom_model("deepseek-chat")
    reasoner = custom_model("deepseek-reasoner")


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
        messages: Optional[List[Dict[str, str]]] = None,
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
                            if chunk.token.text:
                                content = chunk.token.text
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
            messages: Optional[List[Dict[str, str]]] = None,
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
    qwen2_5_coder = custom_model("Qwen/Qwen2.5-Coder-32B-Instruct")
    meta_llama_3_8b = custom_model("meta-llama/Meta-Llama-3-8B-Instruct")
