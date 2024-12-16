# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import time
import random
import re
import threading
import json
from typing import List, Dict, Union, Tuple, Optional, Iterator, AsyncGenerator
from halo import Halo
from anthropic import (
    AsyncAnthropic,
    APIStatusError as AnthropicStatusError,
    APITimeoutError as AnthropicTimeoutError,
    APIConnectionError as AnthropicConnectionError,
    APIResponseValidationError as AnthropicResponseValidationError,
    RateLimitError as AnthropicRateLimitError
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
    BadRequestError as OpenAIBadRequestError
)
from groq import Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
import ollama

# Import config, fall back to environment variables if not found
try:
    from .config import config
except ImportError:
    import os
    
    class EnvConfig:
        def __init__(self):
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
            self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
            self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
            self.TOGETHERAI_API_KEY = os.getenv('TOGETHERAI_API_KEY')
    
    config = EnvConfig()

# Global settings
verbosity = False
debug = False

# Retry settings
MAX_RETRIES = 3
BASE_DELAY = 1
MAX_DELAY = 10

# Define color codes
COLORS = {
    'cyan': '\033[96m',
    'blue': '\033[94m',
    'light_blue': '\033[38;5;39m', 
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'reset': '\033[0m'
}

def set_verbosity(value: Union[str, bool, int]):
    global verbosity, debug
    if isinstance(value, str):
        value = value.lower()
        if value in ['debug', '2']:
            verbosity = True
            debug = True
        elif value in ['true', '1']:
            verbosity = True
            debug = False
        else:
            verbosity = False
            debug = False
    elif isinstance(value, bool):
        verbosity = value
        debug = False
    elif isinstance(value, int):
        if value == 2:
            verbosity = True
            debug = True
        elif value == 1:
            verbosity = True
            debug = False
        else:
            verbosity = False
            debug = False

def print_color(message, color):
    print(f"{COLORS.get(color, '')}{message}{COLORS['reset']}")

def print_conditional_color(message, color):
    if verbosity:
        print_color(message, color)

def print_api_request(message):
    if verbosity:
        print_color(message, 'green')

def print_model_request(provider: str, model: str):
    if verbosity:
        print_color(f"Sending request to {model} from {provider}", 'cyan')

def print_label(message:str):
    if verbosity:
        print_color(message, 'cyan')

def print_api_response(message):
    if verbosity:
        print_color(message, 'blue')

def print_debug(message):
    if debug:
        print_color(message, 'yellow')

def print_error(message):
    print_color(message, 'red')

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
    try:
        # First attempt: Try to parse the entire response
        return json.loads(response)
    except json.JSONDecodeError as e:
        print_color(f"Initial JSON parse failed: {e}", 'yellow')
        
        # Second attempt: Find the first complete JSON object
        json_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
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
        cleaved_json = response.strip().lstrip('`').rstrip('`')
        try:
            return json.loads(cleaved_json)
        except json.JSONDecodeError as e:
            print_error(f"All JSON parsing attempts failed: {e}")
            raise ValueError(f"Invalid JSON structure: {e}")

class OpenaiModels:
    """
    Class containing methods for interacting with OpenAI models.
    """

    @staticmethod
    def _transform_o1_messages(messages: List[Dict[str, str]], require_json_output: bool = False) -> List[Dict[str, str]]:
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
            modified_messages.append({
                "role": "user",
                "content": f"[System Instructions]\n{system_content}"
            })
        
        # Process remaining messages
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                content = msg["content"]
                if require_json_output and msg == messages[-1]:  # If this is the last user message
                    content += "\n\nDo NOT include backticks, language declarations, or commentary before or after the JSON content."
                modified_messages.append({
                    "role": "user",
                    "content": content
                })
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
        stream: bool = False
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
            print_error(f"Streaming is not supported for {model}. Falling back to non-streaming request.")
            stream = False

        spinner = Halo(text='Sending request to OpenAI...', spinner='dots')
        spinner.start()
        
        try:
            api_key = config.validate_api_key("OPENAI_API_KEY")
            client = AsyncOpenAI(api_key=api_key)
            if not client.api_key:
                raise ValueError("OpenAI API key not found in environment variables.")

            # Debug print
            print_conditional_color(f"\n[LLM] OpenAI ({model}) Request Messages:", 'cyan')
            
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

            # Print final messages for debugging
            for msg in messages:
                print_api_request(json.dumps(msg, indent=2))

            if stream:
                spinner.stop()  # Stop spinner before streaming
                async def stream_generator():
                    try:
                        stream_params = {**request_params, "stream": True}
                        response = await client.chat.completions.create(**stream_params)
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                if debug:
                                    print_debug(f"Streaming chunk: {content}")
                                yield content
                    except OpenAIAuthenticationError as e:
                        print_error(f"Authentication failed: Please check your OpenAI API key. Error: {str(e)}")
                        yield ""
                    except OpenAIBadRequestError as e:
                        print_error(f"Invalid request parameters: {str(e)}")
                        yield ""
                    except (OpenAIConnectionError, OpenAITimeoutError) as e:
                        print_error(f"Connection error: {str(e)}")
                        yield ""
                    except OpenAIRateLimitError as e:
                        print_error(f"Rate limit exceeded: {str(e)}")
                        yield ""
                    except OpenAIAPIError as e:
                        print_error(f"OpenAI API error: {str(e)}")
                        yield ""
                    except Exception as e:
                        print_error(f"An unexpected error occurred during streaming: {e}")
                        yield ""

                return stream_generator()

            # Non-streaming logic
            spinner.text = f"Waiting for {model} response..."
            response: OpenAIChatCompletion = await client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            spinner.succeed('Request completed')
            
            if verbosity:
                print_conditional_color("\n[LLM] Actual API Response:", 'light_blue')
                print_api_response(content.strip())
            return content.strip(), None

        except OpenAIAuthenticationError as e:
            spinner.fail('Authentication failed')
            print_error(f"Authentication failed: Please check your OpenAI API key. Error: {str(e)}")
            return "", e
        except OpenAIBadRequestError as e:
            spinner.fail('Invalid request')
            print_error(f"Invalid request parameters: {str(e)}")
            return "", e
        except (OpenAIConnectionError, OpenAITimeoutError) as e:
            spinner.fail('Connection failed')
            print_error(f"Connection error: {str(e)}")
            return "", e
        except OpenAIRateLimitError as e:
            spinner.fail('Rate limit exceeded')
            print_error(f"Rate limit exceeded: {str(e)}")
            return "", e
        except OpenAIAPIError as e:
            spinner.fail('API Error')
            print_error(f"OpenAI API error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail('Request failed')
            print_error(f"Unexpected error: {str(e)}")
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
            stream: bool = False  # Added stream parameter
        ) -> Tuple[str, Optional[Exception]]:
            return await OpenaiModels.send_openai_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream  # Pass stream to send_openai_request
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
        stream: bool = False  # Add stream parameter
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:  # Update return type
        """
        Sends an asynchronous request to an Anthropic model using the Messages API format.
        """
                # Add warning if JSON output is requested
        if require_json_output:
            print_conditional_color(
                "\n[WARNING] JSON output requirement is not directly supported by Anthropic models. "
                "Proceeding with standard output.",
                'yellow'
            )
        spinner = Halo(text='Sending request to Anthropic...', spinner='dots')
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
                    
                    # Handle system messages separately - ISSUE: Currently overwriting with parameter
                    if role == "system":
                        system_message = content  # Store the system message from messages
                    elif role == "user":
                        anthropic_messages.append({
                            "role": "user",
                            "content": content
                        })
                    elif role == "assistant":
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": content
                        })
                    elif role == "function":
                        anthropic_messages.append({
                            "role": "user",
                            "content": f"Function result: {content}"
                        })
                        
            # Handle image data if present
            if image_data:
                if isinstance(image_data, str):
                    image_data = [image_data]
                
                # Add images to the last user message or create new one
                last_msg = anthropic_messages[-1] if anthropic_messages else {"role": "user", "content": []}
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
                    last_msg["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",  # Adjust based on your needs
                            "data": img
                        }
                    })

            # Debug print the request parameters with colors
            print_conditional_color(f"\n[LLM] Anthropic ({model}) Request Messages:", 'cyan')
            print(f"  \033[36msystem_message\033[0m: \033[32m{system_message}\033[0m")  # Print system message
            print("  \033[36mmessages\033[0m:")
            for msg in anthropic_messages:
                print(f"    \033[32m{msg}\033[0m")
            print(f"  \033[36mtemperature\033[0m: \033[32m{temperature}\033[0m") 
            print(f"  \033[36mmax_tokens\033[0m: \033[32m{max_tokens}\033[0m")
            print(f"  \033[36mstop_sequences\033[0m: \033[32m{stop_sequences if stop_sequences else None}\033[0m")

            # Handle streaming
            if stream:
                spinner.stop()  # Stop spinner before streaming
                async def stream_generator():
                    try:
                        response = await client.messages.create(
                            model=model,
                            messages=anthropic_messages,
                            system=system_message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop_sequences=stop_sequences if stop_sequences else None,
                            stream=True
                        )
                        async for chunk in response:
                            # Handle different event types according to Anthropic's streaming format
                            if chunk.type == "content_block_delta":
                                if chunk.delta.type == "text_delta":
                                    content = chunk.delta.text
                                    if debug:
                                        print_debug(f"Streaming chunk: {content}")
                                    yield content
                            elif chunk.type == "message_delta":
                                # Handle message completion
                                if chunk.delta.stop_reason:
                                    if debug:
                                        print_debug(f"Stream completed: {chunk.delta.stop_reason}")
                            elif chunk.type == "error":
                                print_error(f"Stream error: {chunk.error}")
                                break

                    except (AnthropicConnectionError, AnthropicTimeoutError) as e:
                        print_error(f"Connection error during streaming: {str(e)}")
                        yield ""
                    except AnthropicRateLimitError as e:
                        print_error(f"Rate limit exceeded during streaming: {str(e)}")
                        yield ""
                    except AnthropicStatusError as e:
                        print_error(f"API status error during streaming: {str(e)}")
                        yield ""
                    except AnthropicResponseValidationError as e:
                        print_error(f"Invalid response format during streaming: {str(e)}")
                        yield ""
                    except ValueError as e:
                        print_error(f"Configuration error during streaming: {str(e)}")
                        yield ""
                    except Exception as e:
                        print_error(f"An unexpected error occurred during streaming: {e}")
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
                stop_sequences=stop_sequences if stop_sequences else None
            )
            
            content = response.content[0].text if response.content else ""
            spinner.succeed('Request completed')

            if verbosity:
                print_conditional_color("\n[LLM] Actual API Response:", 'light_blue')
                print_api_response(content.strip())

            return content.strip(), None

        except (AnthropicConnectionError, AnthropicTimeoutError) as e:
            spinner.fail('Connection failed')
            print_error(f"Connection error: {str(e)}")
            return "", e
        except AnthropicRateLimitError as e:
            spinner.fail('Rate limit exceeded')
            print_error(f"Rate limit exceeded: {str(e)}")
            return "", e
        except AnthropicStatusError as e:
            spinner.fail('API Status Error')
            print_error(f"API Status Error: {str(e)}")
            return "", e
        except AnthropicResponseValidationError as e:
            spinner.fail('Invalid Response Format')
            print_error(f"Invalid response format: {str(e)}")
            return "", e
        except ValueError as e:
            spinner.fail('Configuration Error')
            print_error(f"Configuration error: {str(e)}")
            return "", e
        except Exception as e:
            spinner.fail('Request failed')
            print_error(f"Unexpected error: {str(e)}")
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
            stream: bool = False  # Add stream parameter
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:  # Update return type
            return await AnthropicModels.send_anthropic_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stop_sequences=stop_sequences,
                stream=stream  # Pass stream parameter
            )
        return wrapper

    # Model-specific methods using custom_model
    opus = custom_model("claude-3-opus-latest")  # or claude-3-opus-20240229
    sonnet = custom_model("claude-3-sonnet-20240229")  # or claude-3-sonnet-20240229 
    haiku = custom_model("claude-3-haiku-20240307")
    sonnet_3_5 = custom_model("claude-3-5-sonnet-latest")  # or claude-3-5-sonnet-20241022
    haiku_3_5 = custom_model("claude-3-5-haiku-latest") 

# Following providers are not yet converted to messages array

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
        stream: bool = False
    ) -> Union[Tuple[str, Optional[Exception]], Iterator[str]]:
        """
        Sends a request to OpenRouter API asynchronously and handles retries.
        """
        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY
            )
            if not client.api_key:
                raise ValueError("OpenRouter API key not found in environment variables.")

            # Debug print
            print_conditional_color(f"\n[LLM] OpenRouter ({model}) Request Messages:", 'cyan')
            for msg in messages:
                print_api_request(json.dumps(msg, indent=2))

            if stream:
                async def stream_generator():
                    try:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                            response_format={"type": "json_object"} if require_json_output else None
                        )
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                if debug:
                                    print_debug(f"Streaming chunk: {content}")
                                yield content
                    except Exception as e:
                        print_error(f"An error occurred during streaming: {e}")
                        yield ""

                return stream_generator()

            # Non-streaming logic
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if require_json_output else None
            )
            
            content = response.choices[0].message.content
            if verbosity:
                print_conditional_color("\n[LLM] Actual API Response:", 'light_blue')
                print_api_response(content.strip())
            return content.strip(), None

        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            return "", e

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, str]]] = None,
            stream: bool = False
        ) -> Tuple[str, Optional[Exception]]:
            return await OpenrouterModels.send_openrouter_request(
                model=model_name,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                messages=messages,
                stream=stream
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
    def call_ollama(model: str, messages: Optional[List[Dict[str, str]]] = None, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        """
        Updated to handle messages array format compatible with Task class.
        """
        print_model_request("Ollama", model)
        if debug:
            print_debug(f"Entering call_ollama function")
            print_debug(f"Parameters: model={model}, messages={messages}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        spinner = Halo(text='Sending request to Ollama...', spinner='dots')
        stop_spinner = threading.Event()
        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            # Process messages into Ollama format
            if not messages:
                messages = []

            # Handle image data by appending to messages
            if image_data:
                print_debug("Processing image data")
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
                    messages.append({
                        "role": "user",
                        "content": image_content
                    })

            print_debug(f"Final messages structure: {messages}")

            for attempt in range(MAX_RETRIES):
                print_debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
                try:
                    client = ollama.Client()
                    print_conditional_color(f"\n[LLM] Ollama ({model}) Request Messages:", 'cyan')
                    for msg in messages:
                        print_api_request(json.dumps(msg, indent=2))
                    response = client.chat(
                        model=model,
                        messages=messages,
                        format="json" if require_json_output else None,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )

                    response_text = response['message']['content']
                    
                    if require_json_output:
                        try:
                            json_response = parse_json_response(response_text)
                        except ValueError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                        
                        return json.dumps(json_response), None
                    
                    return response_text.strip(), None

                except ollama.ResponseError as e:
                    print_error(f"Ollama response error: {e}")
                    print_debug(f"ResponseError details: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except ollama.RequestError as e:
                    print_error(f"Ollama request error: {e}")
                    print_debug(f"RequestError details: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except Exception as e:
                    print_error(f"An unexpected error occurred: {e}")
                    print_debug(f"Unexpected error details: {type(e).__name__}, {e}")
                    return "", e

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')

        return "", Exception("Max retries reached")

    @staticmethod
    def custom_model(model_name: str):
        async def wrapper(
            messages: Optional[List[Dict[str, str]]] = None,
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            stream: bool = False  # Added for compatibility though not used
        ) -> Tuple[str, Optional[Exception]]:
            return OllamaModels.call_ollama(
                model=model_name,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output
            )
        return wrapper

class GroqModels:
    @staticmethod
    def call_groq(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("Groq", model)
        if debug:
            print_debug(f"Entering call_groq function")
            print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: Included")

        spinner = Halo(text='Sending request to Groq...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            for attempt in range(MAX_RETRIES):
                print_debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
                try:
                    api_key = config.GROQ_API_KEY
                    if not api_key:
                        return "", ValueError("GROQ_API_KEY environment variable is not set")

                    print_debug(f"API Key: {api_key[:5]}...{api_key[-5:]}")
                    client = Groq(api_key=api_key)
                    print_debug(f"Groq client initialized")

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    if image_data:
                        print_debug("Processing image data")
                        if isinstance(image_data, str):
                            image_data = [image_data]
                        
                        for i, image in enumerate(image_data, start=1):
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Image {i}:"},
                                    {"type": "image_url", "image_url": {"url": image}}
                                ]
                            })

                    print_conditional_color(f"\n[LLM] Groq ({model}) Request Messages:", 'cyan')
                    for msg in messages:
                        print_api_request(json.dumps(msg, indent=2))

                    response: GroqChatCompletion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if require_json_output else None
                    )

                    print_debug(f"API response received")
                    
                    response_text = response.choices[0].message.content
                    print_debug(f"Processed response text (truncated): {response_text[:100]}...")
                    
                    if require_json_output:
                        try:
                            json_response = parse_json_response(response_text)
                        except ValueError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                        
                        return json.dumps(json_response), None
                    
                    return response_text.strip(), None

                except OpenAIRateLimitError as e:
                    print_error(f"Rate limit exceeded: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_error(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except OpenAITimeoutError as e:
                    print_error(f"API request timed out: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        print_error(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        return "", e

                except OpenAIConnectionError as e:
                    print_error(f"API connection error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        print_error(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        return "", e

                except OpenAIAPIError as e:
                    print_error(f"API error: {e}")
                    return "", e

                except Exception as e:
                    print_error(f"An unexpected error occurred: {e}")
                    print_debug(f"Error details: {type(e).__name__}, {e}")
                    return "", e

            print_debug("Max retries reached")
            return "", Exception("Max retries reached")

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')


    @staticmethod
    def gemma2_9b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "gemma2-9b-it", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemma_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "gemma-7b-it", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_groq_70b_tool_use(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-groq-70b-8192-tool-use-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_groq_8b_tool_use(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-groq-8b-8192-tool-use-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 8000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-3.1-70b-versatile", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 8000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-3.1-8b-instant", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_guard_3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-guard-3-8b", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llava_1_5_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llava-v1.5-7b-4096-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-70b-8192", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-8b-8192", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "mixtral-8x7b-32768", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return GroqModels.call_groq(system_prompt, user_prompt, model_name, image_data, temperature, max_tokens, require_json_output)
        return wrapper

class TogetheraiModels:
    @staticmethod
    def call_together(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        JSON_SUPPORTED_MODELS = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1"
        }

        # Determine if we should use JSON mode
        use_json = require_json_output and model in JSON_SUPPORTED_MODELS
        
        if require_json_output and model not in JSON_SUPPORTED_MODELS:
            print(f"JSON output requested but not supported for model {model}. Falling back to standard output.")

        print_model_request("Together AI", model)
        if debug:
            print_debug(f"Entering call_together function")
            print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: Included")

        spinner = Halo(text='Sending request to Together AI...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            for attempt in range(MAX_RETRIES):
                print_debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
                try:
                    api_key = config.TOGETHERAI_API_KEY
                    if not api_key:
                        return "", ValueError("TOGETHERAI_API_KEY environment variable is not set")

                    print_debug(f"API Key: {api_key[:5]}...{api_key[-5:]}")
                    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
                    print_debug(f"Together client initialized via OpenAI")

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": []}
                    ]

                    if image_data:
                        print_debug("Processing image data")
                        if isinstance(image_data, str):
                            image_data = [image_data]
                        
                        for i, image in enumerate(image_data, start=1):
                            messages[1]["content"].append({"type": "text", "text": f"Image {i}:"})
                            if image.startswith(('http://', 'https://')):
                                print_debug(f"Image {i} is a URL")
                                messages[1]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": image}
                                })
                            else:
                                print_debug(f"Image {i} is base64")
                                messages[1]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                                })
                        
                        messages[1]["content"].append({"type": "text", "text": user_prompt})
                    else:
                        messages[1]["content"] = user_prompt

                    print_conditional_color(f"\n[LLM] TogetherAI ({model}) Request Messages:", 'cyan')
                    for msg in messages:
                        print_api_request(json.dumps(msg, indent=2))

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if use_json else None
                    )

                    print_debug(f"API response received")
                    response_text = response.choices[0].message.content
                    print_debug(f"Processed response text (truncated): {response_text[:100]}...")

                    if require_json_output:
                        try:
                            json_response = parse_json_response(response_text)
                        except ValueError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                        return json.dumps(json_response), None

                    return response_text.strip(), None

                except Exception as e:
                    print_error(f"An error occurred: {e}")
                    print_debug(f"Error details: {type(e).__name__}, {e}")
                    if attempt < MAX_RETRIES - 1:
                        retry_delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

            print_debug("Max retries reached")
            return "", Exception("Max retries reached")

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')

    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return TogetheraiModels.call_together(system_prompt, user_prompt, model_name, image_data, temperature, max_tokens, require_json_output)
        return wrapper
