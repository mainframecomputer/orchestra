"""
Mainframe Orchestra: a Python framework for building and orchestrating multi-agent systems powered by LLMs.
"""
# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

__version__ = "0.0.13"

from .task import Task, configure_logging, LogColors, default_logger
from .agent import Agent
from .config import Config
from .orchestration import Conduct, Compose, TaskInstruction
from .llm import (
    set_verbosity,
    OpenaiModels,
    OpenrouterModels,
    AnthropicModels,
    OllamaModels,
    GroqModels,
    TogetheraiModels,
    GeminiModels
)
from .tools import (
    FileTools,
    EmbeddingsTools,
    WebTools,
    GitHubTools,
    WikipediaTools,
    AmadeusTools,
    CalculatorTools,
    FAISSTools,
    PineconeTools,
    LinearTools,
    SemanticSplitter,
    SentenceSplitter,
    WhisperTools 
)

# Conditional imports for optional dependencies
import sys
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.langchain_tools import LangchainTools
    from .tools.matplotlib_tools import MatplotlibTools
    from .tools.yahoo_finance_tools import YahooFinanceTools
    from .tools.fred_tools import FredTools
    from .tools.audio_tools import WhisperTools, TextToSpeechTools
    from .tools.stripe_tools import StripeTools 

def __getattr__(name):
    package_map = {
        "LangchainTools": (
            "langchain_tools",
            ["langchain-core", "langchain-community", "langchain-openai"],
        ),
        "MatplotlibTools": ("matplotlib_tools", ["matplotlib"]),
        "YahooFinanceTools": ("yahoo_finance_tools", ["yfinance", "yahoofinance"]),
        "FredTools": ("fred_tools", ["fredapi"]),
        "StripeTools": ("stripe_tools", ["stripe", "stripe_agent_toolkit"]),
        "TextToSpeechTools": ("audio_tools", ["elevenlabs", "pygame"]),
    }

    if name in package_map:
        module_name, required_packages = package_map[name]
        try:
            for package in required_packages:
                importlib.import_module(package)

            # If successful, import and return the tool
            module = __import__(f"mainframe_orchestra.tools.{module_name}", fromlist=[name])
            return getattr(module, name)
        except ImportError as e:
            missing_packages = " ".join(required_packages)
            print(
                f"\033[95mError: The required packages ({missing_packages}) are not installed. "
                f"Please install them using 'pip install {missing_packages}'.\n"
                f"Specific error: {str(e)}\033[0m"
            )
            sys.exit(1)
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core Classes
    "Task",
    "Agent",
    "Conduct",
    "Compose",
    "TaskInstruction",
    # Configuration and Utilities
    "Config",
    "config",
    "Utils",
    "set_verbosity",
    # Logging
    "configure_logging",
    "LogColors",
    "default_logger",
    # LLM Provider Models
    "OpenaiModels",
    "AnthropicModels",
    "OpenrouterModels",
    "OllamaModels",
    "GroqModels",
    "TogetheraiModels",
    "GeminiModels",
    # List core tools
    "FileTools",
    "EmbeddingsTools",
    "WebTools",
    "GitHubTools",
    "WikipediaTools",
    "AmadeusTools",
    "CalculatorTools",
    "FAISSTools",
    "PineconeTools",
    "LinearTools",
    "SemanticSplitter",
    "SentenceSplitter",
    "WhisperTools",
    # Optional tools
    "LangchainTools",
    "MatplotlibTools",
    "YahooFinanceTools",
    "TextToSpeechTools",
    "FredTools",
    "StripeTools",
]
