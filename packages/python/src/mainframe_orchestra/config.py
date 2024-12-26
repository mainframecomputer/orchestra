# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
import os
from typing import Optional


class Config(ABC):
    """Base configuration class that can be extended by applications."""

    DEFAULT_TIMEZONE = ZoneInfo("America/New_York")

    # LLM Provider API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    TOGETHERAI_API_KEY: Optional[str] = None

    @classmethod
    def validate_api_key(cls, key_name: str) -> str:
        """Validate and return a specific API key when it's needed."""
        key_value = getattr(cls, key_name, None) or os.getenv(key_name)
        if not key_value:
            raise ValueError(f"{key_name} environment variable is not set")
        return key_value

    @classmethod
    @abstractmethod
    def validate_required_env_vars(cls) -> None:
        """Validate required environment variables for the application."""
        raise NotImplementedError("Subclasses must implement validate_required_env_vars")


# Simple environment-based config for direct use
class EnvConfig(Config):
    """Environment-based configuration that pulls from env vars."""


    def __init__(self):
        # Initialize LLM Provider API Keys from environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")

    @classmethod
    def validate_required_env_vars(cls) -> None:
        pass  # No validation required for basic usage


config = EnvConfig()