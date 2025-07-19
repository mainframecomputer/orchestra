# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

# Core tools
from .amadeus_tools import AmadeusTools
from .audio_tools import WhisperTools
from .calculator_tools import CalculatorTools
from .embedding_tools import EmbeddingsTools
from .faiss_tools import FAISSTools
from .file_tools import FileTools
from .github_tools import GitHubTools
from .linear_tools import LinearTools
from .pinecone_tools import PineconeTools
from .text_splitters import SemanticSplitter, SentenceSplitter
from .web_tools import WebTools
from .wikipedia_tools import WikipediaTools

__all__ = [
    "AmadeusTools",
    "CalculatorTools",
    "EmbeddingsTools",
    "FAISSTools",
    "FileTools",
    "GitHubTools",
    "LinearTools",
    "PineconeTools",
    "WebTools",
    "WikipediaTools",
    "SemanticSplitter",
    "SentenceSplitter",
    "WhisperTools",
]


# Helper function for optional imports
def _optional_import(tool_name, install_name):
    class OptionalTool:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"The tool '{tool_name}' requires additional dependencies. "
                f"Please install them using: 'pip install {install_name}'"
            )

    return OptionalTool


# Conditional imports or placeholders
try:
    from .langchain_tools import LangchainTools

    __all__.append("LangchainTools")
except ImportError:
    LangchainTools = _optional_import("LangchainTools", "langchain_tools")

try:
    from .matplotlib_tools import MatplotlibTools

    __all__.append("MatplotlibTools")
except ImportError:
    MatplotlibTools = _optional_import("MatplotlibTools", "matplotlib_tools")

try:
    from .fred_tools import FredTools

    __all__.append("FredTools")
except ImportError:
    FredTools = _optional_import("FredTools", "fred_tools")

try:
    from .stripe_tools import StripeTools

    __all__.append("StripeTools")
except ImportError:
    StripeTools = _optional_import("StripeTools", "stripe stripe_agent_toolkit")

try:
    from .audio_tools import TextToSpeechTools

    __all__.append("TextToSpeechTools")
except ImportError:
    TextToSpeechTools = _optional_import("TextToSpeechTools", "elevenlabs pygame")
