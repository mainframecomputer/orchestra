"""
Mainframe Orchestra: a Python framework for building and orchestrating multi-agent systems powered by LLMs.
"""
# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

__version__ = "0.0.31"

import importlib

# Conditional imports for optional dependencies
import sys
from typing import TYPE_CHECKING

from .agent import Agent
from .config import Config
from .llm import (
    AnthropicModels,
    DeepseekModels,
    GeminiModels,
    GroqModels,
    HuggingFaceModels,
    OllamaModels,
    OpenaiModels,
    OpenrouterModels,
    TogetheraiModels,
    set_verbosity,
)
from .orchestration import Compose, Conduct, TaskInstruction
from .task import Task
from .tools import (
    AmadeusTools,
    CalculatorTools,
    EmbeddingsTools,
    FAISSTools,
    FileTools,
    GitHubTools,
    LinearTools,
    PineconeTools,
    SemanticSplitter,
    SentenceSplitter,
    WebTools,
    WhisperTools,
    WikipediaTools,
)
from .adapters import MCPOrchestra

if TYPE_CHECKING:
    from .tools.audio_tools import TextToSpeechTools, WhisperTools
    from .tools.fred_tools import FredTools
    from .tools.langchain_tools import LangchainTools
    from .tools.matplotlib_tools import MatplotlibTools
    from .tools.stripe_tools import StripeTools
    from .tools.yahoo_finance_tools import YahooFinanceTools


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
    # LLM Provider Models
    "OpenaiModels",
    "AnthropicModels",
    "OpenrouterModels",
    "OllamaModels",
    "GroqModels",
    "TogetheraiModels",
    "GeminiModels",
    "DeepseekModels",
    "HuggingFaceModels",

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

    # Adapters
    "MCPOrchestra",
]
