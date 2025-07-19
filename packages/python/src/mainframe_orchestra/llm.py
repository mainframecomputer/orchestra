# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import atexit
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import litellm
from litellm import acompletion

# Import the configured logger
from .utils.logging_config import logger

# Import config, fall back to environment variables if not found
try:
    from .config import config
except ImportError:
    import os

    class EnvConfig:
        def __init__(self):
            # OpenAI
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

            # Anthropic
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

            # Other providers
            self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            self.TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
            self.HF_TOKEN = os.getenv("HF_TOKEN")
            self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

        def validate_api_key(self, key_name: str) -> str:
            """Retrieves API key and raises error if missing."""
            api_key = getattr(self, key_name, None)
            if not api_key:
                raise ValueError(f"{key_name} not found in environment variables.")
            return api_key

    config = EnvConfig()

# Enable automatic parameter dropping for unsupported model parameters
litellm.drop_params = True


def set_verbosity(value: Union[str, bool, int]):
    """Set logging verbosity for LiteLLM and mainframe-orchestra."""
    global verbosity, debug

    if isinstance(value, str):
        if value.lower() in ["debug", "2"]:
            verbosity = True
            debug = True
            logger.setLevel(logging.DEBUG)
            os.environ["LITELLM_LOG"] = "DEBUG"
        elif value.lower() in ["info", "verbose", "1"]:
            verbosity = True
            debug = False
            logger.setLevel(logging.INFO)
            os.environ["LITELLM_LOG"] = "INFO"
        else:
            verbosity = False
            debug = False
            logger.setLevel(logging.WARNING)
            os.environ.pop("LITELLM_LOG", None)
    elif isinstance(value, bool):
        verbosity = value
        debug = False
        logger.setLevel(logging.INFO if value else logging.WARNING)
        if value:
            os.environ["LITELLM_LOG"] = "INFO"
        else:
            os.environ.pop("LITELLM_LOG", None)
    elif isinstance(value, int):
        if value == 2:
            verbosity = True
            debug = True
            logger.setLevel(logging.DEBUG)
            os.environ["LITELLM_LOG"] = "DEBUG"
        elif value == 1:
            verbosity = True
            debug = False
            logger.setLevel(logging.INFO)
            os.environ["LITELLM_LOG"] = "INFO"
        else:
            verbosity = False
            debug = False
            logger.setLevel(logging.WARNING)
            os.environ.pop("LITELLM_LOG", None)


class LiteLLMProvider:
    """
    Unified LLM provider using LiteLLM for all model interactions.
    Replaces the 2000+ lines of custom provider code with a simple unified interface.
    """

    @staticmethod
    async def send_request(
        model: Union[str, List[str]],
        messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
        image_data: Union[List[str], str, None] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        require_json_output: bool = False,
        stream: bool = False,
        max_retries: int = 3,
        **kwargs,
    ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
        """
        Universal LLM request handler using LiteLLM.
        Supports all providers with a single interface.
        Supports fallback models and retries.

        Args:
            model: Single model string or list of models for fallback
            max_retries: Number of retries per model before falling back
        """
        # Convert single model to list for unified handling
        models = [model] if isinstance(model, str) else model
        last_exception = None

        for model_name in models:
            logger.debug(f"[LiteLLM] Trying model: {model_name}")

            # Try each model with retries
            for attempt in range(max_retries):
                try:
                    # Handle image data if present
                    if image_data and messages:
                        last_user_msg = next(
                            (msg for msg in reversed(messages) if msg["role"] == "user"), None
                        )
                        if last_user_msg:
                            content = []
                            if isinstance(image_data, str):
                                image_data = [image_data]

                            for image in image_data:
                                if image.startswith(("http://", "https://")):
                                    content.append(
                                        {"type": "image_url", "image_url": {"url": image}}
                                    )
                                else:
                                    # Handle base64 data
                                    if not image.startswith("data:"):
                                        image = f"data:image/jpeg;base64,{image}"
                                    content.append(
                                        {"type": "image_url", "image_url": {"url": image}}
                                    )

                            # Handle the original content properly
                            original_content = last_user_msg["content"]
                            if isinstance(original_content, list):
                                for item in original_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        content.append(item)
                                    elif isinstance(item, str):
                                        content.append({"type": "text", "text": item})
                            elif isinstance(original_content, str):
                                content.append({"type": "text", "text": original_content})
                            elif original_content is not None:
                                content.append({"type": "text", "text": str(original_content)})

                            last_user_msg["content"] = content

                    # Prepare LiteLLM parameters
                    litellm_params = {
                        "model": model_name,
                        "messages": messages or [],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream,
                        **kwargs,
                    }

                    # Handle JSON output
                    if require_json_output:
                        # Different providers handle JSON differently
                        if model_name.startswith(("openai/", "gpt-", "o1-", "o3-", "o4-")):
                            litellm_params["response_format"] = {"type": "json_object"}
                        elif model_name.startswith("anthropic/"):
                            # Anthropic handles JSON via system message
                            json_instruction = "Respond with valid JSON only. Do not include any text before or after the JSON."
                            if messages:
                                system_msg = next(
                                    (msg for msg in messages if msg["role"] == "system"), None
                                )
                                if system_msg:
                                    system_msg["content"] += f"\n\n{json_instruction}"
                                else:
                                    messages.insert(
                                        0, {"role": "system", "content": json_instruction}
                                    )

                    logger.debug(
                        f"[LiteLLM] {model_name} Request (attempt {attempt + 1}): {litellm_params}"
                    )

                    if stream:
                        return LiteLLMProvider._handle_stream(litellm_params)
                    else:
                        response = await acompletion(**litellm_params)
                        content = response.choices[0].message.content or ""

                        logger.debug(
                            f"[LiteLLM] API Response: {' '.join(content.strip().splitlines())}"
                        )
                        return content.strip(), None

                except Exception as e:
                    last_exception = e
                    logger.warning(f"[LiteLLM] {model_name} attempt {attempt + 1} failed: {str(e)}")

                    # Don't retry on certain errors (like authentication failures)
                    if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                        logger.debug(
                            f"[LiteLLM] Authentication error, skipping retries for {model_name}"
                        )
                        break

                    # If this isn't the last attempt, continue retrying
                    if attempt < max_retries - 1:
                        continue

            # If we get here, all retries for this model failed
            logger.warning(
                f"[LiteLLM] All retries failed for {model_name}, trying next model if available"
            )

        # If we get here, all models and retries failed
        logger.error(f"[LiteLLM] All models failed: {[str(m) for m in models]}")
        return "", last_exception

    @staticmethod
    async def _handle_stream(litellm_params: dict) -> AsyncGenerator[str, None]:
        """Handle streaming responses from LiteLLM."""
        full_message = ""
        logger.debug("Stream started")

        try:
            response = await acompletion(**litellm_params)
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content = delta.content
                        full_message += content
                        yield content

            logger.debug("Stream complete")
            logger.debug(f"Full message: {full_message}")

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"

    @staticmethod
    async def cleanup():
        """Clean up LiteLLM sessions to prevent unclosed session warnings."""
        try:
            # Use LiteLLM's proper async client cleanup
            if hasattr(litellm, "close_litellm_async_clients"):
                await litellm.close_litellm_async_clients()
            elif hasattr(litellm, "aclose"):
                await litellm.aclose()
        except Exception:
            pass  # Silent fail - don't break anything

    @staticmethod
    def cleanup_sync():
        """Synchronous cleanup for sync usage."""
        try:
            # Force cleanup of all LiteLLM clients
            import asyncio

            # Try async cleanup first
            try:
                asyncio.run(LiteLLMProvider.cleanup())
            except Exception:
                pass

            # Also try to close any module-level clients
            if hasattr(litellm, "module_level_client") and litellm.module_level_client:
                try:
                    if hasattr(litellm.module_level_client, "close"):
                        litellm.module_level_client.close()
                except Exception:
                    pass

            if hasattr(litellm, "module_level_aclient") and litellm.module_level_aclient:
                try:
                    if hasattr(litellm.module_level_aclient, "close"):
                        asyncio.run(litellm.module_level_aclient.close())
                except Exception:
                    pass

        except Exception:
            pass  # Silent fail - don't break anything


class OpenaiModels:
    """OpenAI models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            base_url: Optional[str] = None,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Handle base_url for custom endpoints
            if base_url:
                kwargs["api_base"] = base_url

            # Use openai/ prefix for LiteLLM
            model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # OpenAI models
    gpt_4_turbo = _create_model("gpt-4-turbo")
    gpt_3_5_turbo = _create_model("gpt-3.5-turbo")
    gpt_4 = _create_model("gpt-4")
    gpt_4o = _create_model("gpt-4o")
    gpt_4o_mini = _create_model("gpt-4o-mini")
    gpt_4_1 = _create_model("gpt-4.1")
    gpt_4_1_mini = _create_model("gpt-4.1-mini")
    gpt_4_1_nano = _create_model("gpt-4.1-nano")
    gpt_4_5_preview = _create_model("gpt-4.5-preview")
    o1_mini = _create_model("o1-mini")
    o1_preview = _create_model("o1-preview")
    o1 = _create_model("o1")
    o3_mini = _create_model("o3-mini")
    o3 = _create_model("o3")
    o4_mini = _create_model("o4-mini")

    @classmethod
    def set_base_url(cls, base_url: str) -> None:
        """Set a default base URL for all OpenAI requests."""
        os.environ["OPENAI_BASE_URL"] = base_url
        logger.info(f"Set default OpenAI base URL to: {base_url}")

    @classmethod
    def custom_model(cls, model_name: str):
        """Create a custom OpenAI model function."""
        return cls._create_model(model_name)


class AnthropicModels:
    """Anthropic models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stop_sequences: Optional[List[str]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Handle stop sequences
            if stop_sequences:
                kwargs["stop"] = stop_sequences

            # Use anthropic/ prefix for LiteLLM
            model = (
                f"anthropic/{model_name}" if not model_name.startswith("anthropic/") else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Anthropic models
    opus = _create_model("claude-3-opus-latest")
    sonnet = _create_model("claude-3-sonnet-20240229")
    haiku = _create_model("claude-3-haiku-20240307")
    sonnet_3_5 = _create_model("claude-3-5-sonnet-latest")
    haiku_3_5 = _create_model("claude-3-5-haiku-latest")
    sonnet_3_7 = _create_model("claude-3-7-sonnet-latest")
    sonnet_4 = _create_model("claude-sonnet-4-20250514")
    opus_4 = _create_model("claude-opus-4-20250514")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Anthropic model function."""
        return AnthropicModels._create_model(model_name)


class OpenrouterModels:
    """OpenRouter models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use openrouter/ prefix for LiteLLM
            model = (
                f"openrouter/{model_name}"
                if not model_name.startswith("openrouter/")
                else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # OpenRouter models (using the provider/model format)
    haiku = _create_model("anthropic/claude-3-haiku")
    haiku_3_5 = _create_model("anthropic/claude-3.5-haiku")
    sonnet = _create_model("anthropic/claude-3-sonnet")
    sonnet_3_5 = _create_model("anthropic/claude-3.5-sonnet")
    opus = _create_model("anthropic/claude-3-opus")
    gpt_3_5_turbo = _create_model("openai/gpt-3.5-turbo")
    gpt_4_turbo = _create_model("openai/gpt-4-turbo")
    gpt_4 = _create_model("openai/gpt-4")
    gpt_4o = _create_model("openai/gpt-4o")
    gpt_4o_mini = _create_model("openai/gpt-4o-mini")
    o1_preview = _create_model("openai/o1-preview")
    o1_mini = _create_model("openai/o1-mini")
    o3 = _create_model("openai/o3")
    o3_mini = _create_model("openai/o3-mini")
    o4_mini = _create_model("openai/o4-mini")
    gpt_4_1 = _create_model("openai/gpt-4.1")
    gpt_4_1_mini = _create_model("openai/gpt-4.1-mini")
    gpt_4_1_nano = _create_model("openai/gpt-4.1-nano")
    gemini_flash_1_5 = _create_model("google/gemini-flash-1.5")
    llama_3_70b_sonar_32k = _create_model("perplexity/llama-3-sonar-large-32k-chat")
    command_r = _create_model("cohere/command-r-plus")
    nous_hermes_2_mistral_7b_dpo = _create_model("nousresearch/nous-hermes-2-mistral-7b-dpo")
    nous_hermes_2_mixtral_8x7b_dpo = _create_model("nousresearch/nous-hermes-2-mixtral-8x7b-dpo")
    nous_hermes_yi_34b = _create_model("nousresearch/nous-hermes-yi-34b")
    qwen_2_72b = _create_model("qwen/qwen-2-72b-instruct")
    mistral_7b = _create_model("mistralai/mistral-7b-instruct")
    mistral_7b_nitro = _create_model("mistralai/mistral-7b-instruct:nitro")
    mixtral_8x7b_instruct = _create_model("mistralai/mixtral-8x7b-instruct")
    mixtral_8x7b_instruct_nitro = _create_model("mistralai/mixtral-8x7b-instruct:nitro")
    mixtral_8x22b_instruct = _create_model("mistralai/mixtral-8x22b-instruct")
    wizardlm_2_8x22b = _create_model("microsoft/wizardlm-2-8x22b")
    neural_chat_7b = _create_model("intel/neural-chat-7b")
    gemma_7b_it = _create_model("google/gemma-7b-it")
    gemini_pro = _create_model("google/gemini-pro")
    llama_3_8b_instruct = _create_model("meta-llama/llama-3-8b-instruct")
    llama_3_70b_instruct = _create_model("meta-llama/llama-3-70b-instruct")
    llama_3_70b_instruct_nitro = _create_model("meta-llama/llama-3-70b-instruct:nitro")
    llama_3_8b_instruct_nitro = _create_model("meta-llama/llama-3-8b-instruct:nitro")
    dbrx_132b_instruct = _create_model("databricks/dbrx-instruct")
    deepseek_coder = _create_model("deepseek/deepseek-coder")
    llama_3_1_70b_instruct = _create_model("meta-llama/llama-3.1-70b-instruct")
    llama_3_1_8b_instruct = _create_model("meta-llama/llama-3.1-8b-instruct")
    llama_3_1_405b_instruct = _create_model("meta-llama/llama-3.1-405b-instruct")
    qwen_2_5_coder_32b_instruct = _create_model("qwen/qwen-2.5-coder-32b-instruct")
    claude_3_5_haiku = _create_model("anthropic/claude-3-5-haiku")
    ministral_8b = _create_model("mistralai/ministral-8b")
    ministral_3b = _create_model("mistralai/ministral-3b")
    llama_3_1_nemotron_70b_instruct = _create_model("nvidia/llama-3.1-nemotron-70b-instruct")
    gemini_flash_1_5_8b = _create_model("google/gemini-flash-1.5-8b")
    llama_3_2_3b_instruct = _create_model("meta-llama/llama-3.2-3b-instruct")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom OpenRouter model function."""
        return OpenrouterModels._create_model(model_name)


class GroqModels:
    """Groq models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use groq/ prefix for LiteLLM
            model = f"groq/{model_name}" if not model_name.startswith("groq/") else model_name

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Groq models
    gemma2_9b_it = _create_model("gemma2-9b-it")
    llama_3_3_70b_versatile = _create_model("llama-3.3-70b-versatile")
    llama_3_1_8b_instant = _create_model("llama-3.1-8b-instant")
    llama3_70b_8192 = _create_model("llama3-70b-8192")
    llama3_8b_8192 = _create_model("llama3-8b-8192")
    mixtral_8x7b_32768 = _create_model("mixtral-8x7b-32768")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Groq model function."""
        return GroqModels._create_model(model_name)


class DeepseekModels:
    """DeepSeek models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use deepseek/ prefix for LiteLLM
            model = (
                f"deepseek/{model_name}" if not model_name.startswith("deepseek/") else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # DeepSeek models
    chat = _create_model("deepseek-chat")
    reasoner = _create_model("deepseek-reasoner")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom DeepSeek model function."""
        return DeepseekModels._create_model(model_name)


class TogetheraiModels:
    """Together AI models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use together_ai/ prefix for LiteLLM
            model = (
                f"together_ai/{model_name}"
                if not model_name.startswith("together_ai/")
                else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Together AI models
    meta_llama_3_1_70b_instruct_turbo = _create_model(
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    )

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Together AI model function."""
        return TogetheraiModels._create_model(model_name)


class GeminiModels:
    """Google Gemini models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use vertex_ai/ prefix for LiteLLM (Gemini via Vertex AI)
            model = (
                f"vertex_ai/{model_name}" if not model_name.startswith("vertex_ai/") else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Gemini models
    gemini_1_5_flash = _create_model("gemini-1.5-flash")
    gemini_1_5_pro = _create_model("gemini-1.5-pro")
    gemini_2_0_flash = _create_model("gemini-2.0-flash")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Gemini model function."""
        return GeminiModels._create_model(model_name)


class OllamaModels:
    """Ollama models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use ollama/ prefix for LiteLLM
            model = f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Ollama model function."""
        return OllamaModels._create_model(model_name)


class HuggingFaceModels:
    """HuggingFace models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use huggingface/ prefix for LiteLLM
            model = (
                f"huggingface/{model_name}"
                if not model_name.startswith("huggingface/")
                else model_name
            )

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Common HuggingFace models
    qwen2_5_coder = _create_model("Qwen/Qwen2.5-Coder-32B-Instruct")
    meta_llama_3_8b = _create_model("meta-llama/Meta-Llama-3-8B-Instruct")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom HuggingFace model function."""
        return HuggingFaceModels._create_model(model_name)


class XAIModels:
    """xAI models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use xai/ prefix for LiteLLM
            model = f"xai/{model_name}" if not model_name.startswith("xai/") else model_name

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # xAI models
    grok_2_latest = _create_model("grok-2-latest")
    grok_2_vision = _create_model("grok-2-vision")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom xAI model function."""
        return XAIModels._create_model(model_name)


class CohereModels:
    """Cohere models using LiteLLM."""

    @staticmethod
    def _create_model(model_name: str):
        async def wrapper(
            image_data: Union[List[str], str, None] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
            require_json_output: bool = False,
            messages: Optional[List[Dict[str, Union[str, List[Dict[str, Any]]]]]] = None,
            stream: bool = False,
            **kwargs,
        ) -> Union[Tuple[str, Optional[Exception]], AsyncGenerator[str, None]]:
            # Use cohere/ prefix for LiteLLM
            model = f"cohere/{model_name}" if not model_name.startswith("cohere/") else model_name

            return await LiteLLMProvider.send_request(
                model=model,
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json_output=require_json_output,
                stream=stream,
                **kwargs,
            )

        return wrapper

    # Cohere models
    command_r_plus = _create_model("command-r-plus")
    command_r = _create_model("command-r")

    @staticmethod
    def custom_model(model_name: str):
        """Create a custom Cohere model function."""
        return CohereModels._create_model(model_name)


__all__ = [
    "set_verbosity",
    "LiteLLMProvider",
    "OpenaiModels",
    "AnthropicModels",
    "OpenrouterModels",
    "GroqModels",
    "DeepseekModels",
    "TogetheraiModels",
    "GeminiModels",
    "OllamaModels",
    "HuggingFaceModels",
    "XAIModels",
    "CohereModels",
]


# Automatic cleanup on exit
def _cleanup_litellm():
    """Automatically clean up LiteLLM sessions on program exit."""
    try:
        # Use the comprehensive cleanup methods
        LiteLLMProvider.cleanup_sync()

    except Exception:
        pass  # Silent fail during exit


# Register cleanup to run automatically when Python exits
atexit.register(_cleanup_litellm)
