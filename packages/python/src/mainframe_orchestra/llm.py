# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import time
import random
import re
import json
import logging
from typing import List, Dict, Union, Tuple, Optional, Iterator, AsyncGenerator
from halo import Halo
from anthropic import (
    AsyncAnthropic,
    APIStatusError as AnthropicStatusError,
    APITimeoutError as AnthropicTimeoutError,
    APIConnectionError as AnthropicConnectionError,
    APIResponseValidationError as AnthropicResponseValidationError,
    RateLimitError as AnthropicRateLimitError,
)
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai import (
    OpenAI,
    AsyncOpenAI,
    APIError as OpenAIAPIError,
    APIConnectionError as OpenAIConnectionError,
    APITimeoutError as OpenAITimeoutError,
    RateLimitError as OpenAIRateLimitError,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError as OpenAIBadRequestError,
)
from groq import Groq
import ollama
import google.generativeai as genai
from .utils.braintrust_utils import wrap_openai

# Import config, fall back to environment variables if not found
try:
    from .config import config
except ImportError:
    import os

    class EnvConfig:
        def __init__(self):
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            self.TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    config = EnvConfig()

# Global settings
verbosity = False
debug = False

# Retry settings
MAX_RETRIES = 3
BASE_DELAY = 1
MAX_DELAY = 10

# Setup logging
logger = logging.getLogger("mainframe-orchestra")
log_level = os.getenv("ORCHESTRA_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Configure third-party loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)

# Ensure we have at least a console handler if none exists
if not logger.handlers:
    # Console handler - keep it clean for user output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Setup file logging if ORCHESTRA_LOG_FILE is set
    log_file = os.getenv("ORCHESTRA_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        file_handler.setLevel(logging.DEBUG)  # File logs capture everything
        logger.addHandler(file_handler)

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
    # First attempt: Try to parse the entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Second attempt: Find the first complete JSON object
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

        # Third attempt: Try to cleave strings before and after JSON
        cleaved_json = response.strip().lstrip("`").rstrip("`")
        try:
            return json.loads(cleaved_json)
        except json.JSONDecodeError as e:
            logger.warning(f"All JSON parsing attempts failed: {e}")
            raise ValueError(f"Invalid JSON structure: {e}")


class OpenaiModels:
    """
    Class containing methods for interacting with OpenAI models.
    """

    @staticmethod
    def _transform_o1_messages(
        messages: List[Dict[str, str]], require_json_output: bool = False
    ) -> List[Dict[str, str]]:
        """
        Transform messages for o1 models by handling system messages and JSON requirements.

        Args:
            messages (List[Dict[str, str]]): Original messages array
            require_json_output (bool): Whether JSON output is required

        Returns:
            List[Dict[str, str]]: Modified messages array for o1 models
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

    @staticmethod
    async def send_openai_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
        """
        Sends a request to an OpenAI model asynchronously and handles retries.

        Args:
            model (str): The model name.
            image_data (Union[List[str], str, None], optional): Image data if any.
            temperature (float, optional): Sampling temperature.
            max_tokens (int, optional): Maximum tokens in the response.
            require_json_output (bool, optional): If True, requests JSON output.
            messages (List[Dict[str, str]], optional): Direct messages to send to the API.
            stream (bool, optional): If True, enables streaming of responses.

        Returns:
            Union[Tuple[str, Optional[Exception]], Iterator[str]]: The response text and any exception encountered, or an iterator for streaming.
        """

        # Add check for non-streaming models (currently only o1 models) at the start
        if stream and model in ["o1-mini", "o1-preview"]:
            logger.error(f"Streaming is not supported for {model}. Falling back to non-streaming request.")
            stream = False

        spinner = Halo(text="Sending request to OpenAI...", spinner="dots")
        spinner.start()

        try:
            api_key = config.validate_api_key("OPENAI_API_KEY")
            client = wrap_openai(AsyncOpenAI(api_key=api_key))
            if not client.api_key:
                raise ValueError("OpenAI API key not found in environment variables.")

            # Handle all o1-specific modifications
            if model in ["o1-mini", "o1-preview"]:
                messages = OpenaiModels._transform_o1_messages(messages, require_json_output)
                request_params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
            else:
                request_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if require_json_output:
                    request_params["response_format"] = {"type": "json_object"}

            # Log all request details including parameters
            logger.debug(f"[LLM] OpenAI ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

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
                        logger.debug(f"Stream complete: {full_message}")
                    except OpenAIAuthenticationError as e:
                        logger.error(f"Authentication failed: Please check your OpenAI API key. Error: {str(e)}")
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
                        logger.error(f"OpenAI API error: {str(e)}")
                        yield ""
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during streaming: {e}")
                        yield ""

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response: OpenAIChatCompletion = await client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            spinner.succeed("Request completed")

            try:
                # Attempt to parse the API response as JSON and reformat it as compact, single-line JSON.
                compact_response = json.dumps(json.loads(content.strip()), separators=(',', ':'))
            except ValueError:
                # If it's not JSON, preserve newlines but clean up extra whitespace within lines
                lines = content.strip().splitlines()
                compact_response = "\n".join(line.strip() for line in lines)

            logger.debug(f"API Response: {compact_response}")
            return compact_response, None

        except OpenAIAuthenticationError as e:
            spinner.fail("Authentication failed")
            logger.error(f"Authentication failed: Please check your OpenAI API key. Error: {str(e)}")
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
            logger.error(f"OpenAI API error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}")
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
            stream: bool = False,
        ) -> Tuple[str, Optional[Exception]]:
            return await OpenaiModels.send_openai_request(
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
    gpt_4_turbo = custom_model("gpt-4-turbo")
    gpt_3_5_turbo = custom_model("gpt-3.5-turbo")
    gpt_4 = custom_model("gpt-4")
    gpt_4o = custom_model("gpt-4o")
    gpt_4o_mini = custom_model("gpt-4o-mini")
    o1_mini = custom_model("o1-mini")
    o1_preview = custom_model("o1-preview")


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
                    last_msg["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Adjust based on your needs
                                "data": img,
                            },
                        }
                    )
            # Log request details
            logger.debug(f"[LLM] Anthropic ({model}) Request: {json.dumps({'system_message': system_message, 'messages': anthropic_messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stop_sequences': stop_sequences}, separators=(',', ':'))}")

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
                                    logger.debug(f"Message delta stop reason: {chunk.delta.stop_reason}")
                            elif chunk.type == "error":
                                logger.error(f"Stream error: {chunk.error}")
                                break
                        logger.debug("Stream complete")
                        logger.debug(f"Final message: {full_message}")
                    except (AnthropicConnectionError, AnthropicTimeoutError) as e:
                        logger.error(f"Connection error during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except AnthropicRateLimitError as e:
                        logger.error(f"Rate limit exceeded during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except AnthropicStatusError as e:
                        logger.error(f"API status error during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except AnthropicResponseValidationError as e:
                        logger.error(f"Invalid response format during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except ValueError as e:
                        logger.error(f"Configuration error during streaming: {str(e)}", exc_info=True)
                        yield ""
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during streaming: {e}", exc_info=True)
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

    @staticmethod
    async def send_openrouter_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
        """
        Sends a request to OpenRouter API asynchronously and handles retries.
        """
        spinner = Halo(text="Sending request to OpenRouter...", spinner="dots")
        spinner.start()

        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=config.OPENROUTER_API_KEY
            )
            if not client.api_key:
                raise ValueError("OpenRouter API key not found in environment variables.")

            # Log request details including parameters
            logger.debug(f"[LLM] OpenRouter ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")
                    try:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                            response_format={"type": "json_object"} if require_json_output else None,
                        )
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_message += content
                                yield content
                        logger.debug("Stream complete")
                        logger.debug(f"Full message: {full_message}")
                        yield "\n"
                    except Exception as e:
                        logger.error(f"An error occurred during streaming: {e}", exc_info=True)
                        yield "\n"

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if require_json_output else None,
            )

            content = response.choices[0].message.content
            spinner.succeed("Request completed")

            # Compress the response to single line if it's JSON
            if require_json_output:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(',', ':'))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    return "", e
            
            # For non-JSON responses, keep original formatting but make single line
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

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
    opus = custom_model("anthropic/claude-3-opus")
    gpt_3_5_turbo = custom_model("openai/gpt-3.5-turbo")
    gpt_4_turbo = custom_model("openai/gpt-4-turbo")
    gpt_4 = custom_model("openai/gpt-4")
    gpt_4o = custom_model("openai/gpt-4o")
    gpt_4o_mini = custom_model("openai/gpt-4o-mini")
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

                    logger.debug(f"[LLM] Ollama ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

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
                                    if chunk and "message" in chunk and "content" in chunk["message"]:
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
                    logger.debug(f"[LLM] API Response: {' '.join(response_text.strip().splitlines())}")
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
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:  # Update return type
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


class GroqModels:
    @staticmethod
    async def send_groq_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to Groq using the messages API format.
        """
        spinner = Halo(text="Sending request to Groq...", spinner="dots")
        spinner.start()

        try:
            api_key = config.validate_api_key("GROQ_API_KEY")
            client = Groq(api_key=api_key)
            if not client.api_key:
                raise ValueError("Groq API key not found in environment variables.")

            # Log request messages at DEBUG level
            logger.debug(f"[LLM] Groq ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"} if require_json_output else None,
                            stream=True,
                        )
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_message += content
                                yield content
                        logger.debug("Stream complete")
                        logger.debug(f"Full message: {full_message}")
                    except Exception as e:
                        logger.error(f"An error occurred during streaming: {e}")
                        yield ""

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if require_json_output else None,
            )

            content = response.choices[0].message.content
            spinner.succeed("Request completed")

            # Compress the response to single line if it's JSON
            if require_json_output:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(',', ':'))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    return "", e
            
            # For non-JSON responses, keep original formatting but make single line
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}")
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

    gemma2_9b_it = custom_model("gemma2-9b-it")
    llama_3_3_70b_versatile = custom_model("llama-3.3-70b-versatile")
    llama_3_1_8b_instant = custom_model("llama-3.1-8b-instant")
    llama_guard_3_8b = custom_model("llama-guard-3-8b")
    llama3_70b_8192 = custom_model("llama3-70b-8192")
    llama3_8b_8192 = custom_model("llama3-8b-8192")
    mixtral_8x7b_32768 = custom_model("mixtral-8x7b-32768")


class TogetheraiModels:
    @staticmethod
    async def send_together_request(
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
        spinner = Halo(text="Sending request to Together AI...", spinner="dots")
        spinner.start()

        try:
            api_key = config.validate_api_key("TOGETHERAI_API_KEY")
            client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

            # Process images if present
            if image_data:
                last_user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
                if last_user_msg:
                    content = []
                    if isinstance(image_data, str):
                        image_data = [image_data]
                    
                    for i, image in enumerate(image_data, start=1):
                        content.append({"type": "text", "text": f"Image {i}:"})
                        if image.startswith(("http://", "https://")):
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image}
                            })
                        else:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                            })
                    
                    # Add original text content
                    content.append({"type": "text", "text": last_user_msg["content"]})
                    last_user_msg["content"] = content

            # Log request details after any message modifications
            logger.debug(f"[LLM] TogetherAI ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

            if stream:
                spinner.stop()

                async def stream_generator():
                    full_message = ""
                    logger.debug("Stream started")
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"} if require_json_output else None,
                            stream=True
                        )
                        
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_message += content
                                yield content
                        logger.debug("Stream complete")
                        logger.debug(f"Full message: {full_message}")
                        yield "\n" 
                    except Exception as e:
                        logger.error(f"An error occurred during streaming: {e}")
                        yield ""
                        yield "\n" 

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if require_json_output else None,
            )

            content = response.choices[0].message.content
            spinner.succeed("Request completed")
            
            # Compress the response to single line if it's JSON
            if require_json_output:
                try:
                    json_response = parse_json_response(content)
                    compressed_content = json.dumps(json_response, separators=(',', ':'))
                    logger.debug(f"[LLM] API Response: {compressed_content}")
                    return compressed_content, None
                except ValueError as e:
                    return "", e
            
            # For non-JSON responses, keep original formatting but make single line
            logger.debug(f"[LLM] API Response: {' '.join(content.strip().splitlines())}")
            return content.strip(), None

        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}")
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

    meta_llama_3_1_70b_instruct_turbo = custom_model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")


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
        # Create spinner only once at the start
        spinner = Halo(text=f"Sending request to Gemini ({model})...", spinner="dots")
        
        try:
            # Start spinner
            spinner.start()

            # Configure API and model
            api_key = config.validate_api_key("GEMINI_API_KEY")
            genai.configure(api_key=api_key)

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            if require_json_output:
                generation_config.update({
                    "response_mime_type": "application/json"
                })

            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.GenerationConfig(**generation_config)
            )
            # Print all messages together after spinner starts
            logger.debug(f"[LLM] Gemini ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

            if stream:
                spinner.stop()
                last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
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
                                if isinstance(image_data, str):
                                    image_data = [image_data]
                                for img in image_data:
                                    parts.append({"mime_type": "image/jpeg", "data": img})
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
    def _preprocess_reasoner_messages(messages: List[Dict[str, str]], require_json_output: bool = False) -> List[Dict[str, str]]:
        """
        Preprocess messages specifically for the DeepSeek Reasoner model:
        - Combine successive user messages
        - Combine successive assistant messages
        - Handle JSON output requirements

        Args:
            messages (List[Dict[str, str]]): Original messages array
            require_json_output (bool): Whether JSON output was requested

        Returns:
            List[Dict[str, str]]: Processed messages array
        """
        if require_json_output:
            logger.warning("Warning: JSON output format is not supported for the Reasoner model. Request will proceed without JSON formatting.")

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
                    processed.append({
                        "role": current_role,
                        "content": "\n".join(current_content)
                    })
                # Start new message
                current_role = msg["role"]
                current_content = [msg["content"]]

        if current_role:
            processed.append({
                "role": current_role,
                "content": "\n".join(current_content)
            })

        return processed

    @staticmethod
    async def send_deepseek_request(
        model: str = "",
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Sends a request to DeepSeek models asynchronously.
        For the Reasoner model, returns both reasoning and answer as a tuple
        """
        spinner = Halo(text="Sending request to DeepSeek...", spinner="dots")
        spinner.start()

        try:
            # Validate and retrieve the DeepSeek API key
            api_key = config.validate_api_key("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key not found in environment variables.")

            # Create an AsyncOpenAI client
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
            )

            # Warn if image data was provided
            if image_data:
                logger.warning("Warning: DeepSeek API does not support image inputs. Images will be ignored.")

            # Preprocess messages only for the reasoner model
            if messages and model == "deepseek-reasoner":
                messages = DeepseekModels._preprocess_reasoner_messages(messages, require_json_output)
                # Remove JSON requirement for reasoner model
                require_json_output = False
            # Log request details
            logger.debug(f"[LLM] DeepSeek ({model}) Request: {json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'require_json_output': require_json_output, 'stream': stream}, separators=(',', ':'))}")

            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add JSON output format if required (now only for non-reasoner models)
            if require_json_output:
                request_params["response_format"] = {"type": "json_object"}
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\nPlease ensure the response is valid JSON."

            if stream:
                spinner.stop()  # Stop spinner before streaming

                async def stream_generator():
                    if model == "deepseek-reasoner":
                        full_reasoning = ""
                        full_answer = ""
                        in_reasoning = False
                        logger.debug("Stream started")
                        try:
                            response = await client.chat.completions.create(stream=True, **request_params)
                            async for chunk in response:
                                if chunk.choices[0].delta.reasoning_content:
                                    if not in_reasoning:
                                        in_reasoning = True
                                    content = chunk.choices[0].delta.reasoning_content
                                    full_reasoning += content
                                    yield content
                                elif chunk.choices[0].delta.content:
                                    if in_reasoning:
                                        in_reasoning = False
                                    content = chunk.choices[0].delta.content
                                    full_answer += content
                                    yield content
                            logger.debug(f"Stream complete: reasoning: {full_reasoning}, answer: {full_answer}")
                        except Exception as e:
                            logger.error(f"An error occurred during streaming: {e}")
                            yield ""
                    else:
                        full_message = ""
                        logger.debug("Stream started")
                        try:
                            response = await client.chat.completions.create(stream=True, **request_params)
                            async for chunk in response:
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    full_message += content
                                    yield content
                            logger.debug("Stream complete")
                            logger.debug(f"Full message: {full_message}")
                        except Exception as e:
                            logger.error(f"An error occurred during streaming: {e}")
                            yield ""

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response = await client.chat.completions.create(**request_params)
            
            if model == "deepseek-reasoner":
                reasoning = response.choices[0].message.reasoning_content
                content = response.choices[0].message.content
                # Instead of returning a formatted string, compress the output inline
                spinner.succeed("Request completed")
                compressed_reasoning = " ".join(reasoning.strip().split())
                compressed_answer = " ".join(content.strip().split())
                logger.debug(f"[LLM] API Response (Reasoning): {compressed_reasoning}")
                logger.debug(f"[LLM] API Response (Answer): {compressed_answer}")

                return (compressed_reasoning, compressed_answer), None  # Return tuple of (reasoning, answer)
            else:
                content = response.choices[0].message.content
                spinner.succeed("Request completed")
                compressed_content = " ".join(content.strip().split())
                logger.debug(f"[LLM] API Response: {compressed_content}")

                if require_json_output:
                    try:
                        return json.dumps(parse_json_response(content)), None
                    except ValueError as e:
                        logger.error(f"Failed to parse response as JSON: {e}")
                        return "", e

                return compressed_content, None

        except Exception as e:
            spinner.fail("Request failed")
            logger.error(f"Unexpected error: {str(e)}")
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
    
    # Model-specific methods using custom_model
    chat = custom_model("deepseek-chat")
    reasoner = custom_model("deepseek-reasoner")
