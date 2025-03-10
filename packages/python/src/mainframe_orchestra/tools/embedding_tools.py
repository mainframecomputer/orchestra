# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import List, Dict, Union, Literal, Tuple
from dotenv import load_dotenv
import requests
import time
from ..utils.braintrust_utils import traced

# Load environment variables
load_dotenv()

class EmbeddingsTools:
    """
    A class for generating embeddings using various models.
    """
    MODEL_DIMENSIONS = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # Cohere
        "embed-english-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384,
        "embed-multilingual-v2.0": 768,
        # Mistral
        "mistral-embed": 1024
    }

    @traced(type="tool")
    @staticmethod
    def get_model_dimension(provider: str, model: str) -> int:
        """
        Get the dimension of the specified embedding model.

        Args:
            provider (str): The provider of the embedding model.
            model (str): The name of the embedding model.

        Returns:
            int: The dimension of the embedding model.

        Raises:
            ValueError: If the provider or model is not supported.
        """
        if provider == "openai" or provider == "cohere" or provider == "mistral":
            if model in EmbeddingsTools.MODEL_DIMENSIONS:
                return EmbeddingsTools.MODEL_DIMENSIONS[model]

        raise ValueError(f"Unsupported embedding model: {model} for provider: {provider}")

    @traced(type="tool")
    @staticmethod
    def get_openai_embeddings(
        input_text: Union[str, List[str]],
        model: Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] = "text-embedding-3-small"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using OpenAI's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]):
                The model to use for generating embeddings. Default is "text-embedding-3-small".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.exceptions.RequestException: If there's an error with the HTTP request.

        Note:
            This method requires a valid OpenAI API key to be set in the OPENAI_API_KEY environment variable.
        """

        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Prepare the API request
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Ensure input_text is a list and not empty
        if isinstance(input_text, str):
            input_text = [input_text]

        if not input_text or any(not text.strip() for text in input_text):
            raise ValueError("Input text cannot be empty")

        # Validate input length
        max_tokens = 8191  # OpenAI's max token limit
        if any(len(text) > max_tokens for text in input_text):
            raise ValueError(f"Input text exceeds maximum token limit of {max_tokens}")

        payload = {
            "input": input_text,
            "model": model,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            embeddings = [item['embedding'] for item in data['data']]

            return embeddings, {"dimensions": EmbeddingsTools.MODEL_DIMENSIONS[model]}

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error making request to OpenAI API: {str(e)}")

    @traced(type="tool")
    @staticmethod
    def get_cohere_embeddings(
        input_text: Union[str, List[str]],
        model: str = "embed-english-v3.0",
        input_type: str = "search_document"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using Cohere's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (str): The model to use for generating embeddings. Default is "embed-english-v3.0".
            input_type (str): The type of input. Default is "search_document".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set.
            RuntimeError: If there's an error in generating embeddings from the Cohere API.
        """
        # Get API key from environment variable
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        # Check for cohere
        try:
            import cohere
        except ModuleNotFoundError:
            raise ImportError("cohere package is required for Cohere embedding tools. Install with `pip install cohere`")
        cohere_client = cohere.Client(api_key)

        # Ensure input_text is a list
        if isinstance(input_text, str):
            input_text = [input_text]

        try:
            time.sleep(1)  # Rate limiting
            response = cohere_client.embed(
                texts=input_text,
                model=model,
                input_type=input_type
            )
            embeddings = response.embeddings
            return embeddings, {"dimensions": EmbeddingsTools.MODEL_DIMENSIONS[model]}

        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings from Cohere API: {str(e)}")

    @traced(type="tool")
    @staticmethod
    def get_mistral_embeddings(
        input_text: Union[str, List[str]],
        model: str = "mistral-embed"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using Mistral AI's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (str): The model to use for generating embeddings. Default is "mistral-embed".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.exceptions.RequestException: If there's an error with the HTTP request.

        Note:
            This method requires a valid Mistral AI API key to be set in the MISTRAL_API_KEY environment variable.
        """

        # Get API key from environment variable
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        # Prepare the API request
        url = "https://api.mistral.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Ensure input_text is a list
        if isinstance(input_text, str):
            input_text = [input_text]

        payload = {
            "model": model,
            "input": input_text
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            embeddings = [item['embedding'] for item in data['data']]
            dimensions = len(embeddings[0]) if embeddings else 0

            return embeddings, {"dimensions": dimensions}

        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                error_details = e.response.text
            else:
                error_details = str(e)
            raise requests.exceptions.RequestException(f"Error making request to Mistral AI API: {error_details}")

    @traced(type="tool")
    @staticmethod
    def get_embeddings(input_text: Union[str, List[str]], provider: str, model: str) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using the specified provider and model.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            provider (str): The provider to use for generating embeddings.
            model (str): The model to use for generating embeddings.

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the provider or model is not supported.
        """
        if provider == "openai":
            if model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
                return EmbeddingsTools.get_openai_embeddings(input_text, model)
            else:
                raise ValueError(f"Unsupported OpenAI embedding model: {model}")
        elif provider == "cohere":
            if model in [
                "embed-english-v3.0", "embed-english-light-v3.0", "embed-english-v2.0",
                "embed-english-light-v2.0", "embed-multilingual-v3.0", "embed-multilingual-light-v3.0",
                "embed-multilingual-v2.0"
            ]:
                return EmbeddingsTools.get_cohere_embeddings(input_text, model)
            else:
                raise ValueError(f"Unsupported Cohere embedding model: {model}")
        elif provider == "mistral":
            if model == "mistral-embed":
                return EmbeddingsTools.get_mistral_embeddings(input_text, model)
            else:
                raise ValueError(f"Unsupported Mistral embedding model: {model}")
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

