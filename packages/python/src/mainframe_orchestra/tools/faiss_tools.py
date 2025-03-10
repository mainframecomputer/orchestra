# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import json
import numpy as np
from typing import Tuple, Any
from ..utils.braintrust_utils import traced

class FAISSTools:
    def __init__(self, dimension: int, metric: str = "IP"):
        """
        Initialize FAISSTools with the specified dimension and metric.

        Args:
            dimension (int): Dimension of the vectors to be stored in the index.
            metric (str, optional): Distance metric to use. Defaults to "IP" (Inner Product).
        """
        # Load the faiss library
        try:
            import faiss
        except ModuleNotFoundError:
            raise ImportError("faiss is required for FAISSTools. Install with `pip install faiss-cpu` or `pip install faiss-gpu`")

        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.embedding_model = None
        self.embedding_provider = None
        self.metadata = {}  # Added to store metadata

        self.faiss = faiss

    @traced(type="tool")
    def create_index(self, index_type: str = "Flat") -> None:
        """
        Create a new FAISS index.

        Args:
            index_type (str, optional): Type of index to create. Defaults to "Flat".

        Raises:
            ValueError: If an unsupported index type is specified.
        """
        if index_type == "Flat":
            if self.metric == "IP":
                self.index = self.faiss.IndexFlatIP(self.dimension)
            elif self.metric == "L2":
                self.index = self.faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    @traced(type="tool")
    def load_index(self, index_path: str) -> None:
        """
        Load a FAISS index and metadata from files.

        Args:
            index_path (str): Path to the index file.

        Raises:
            FileNotFoundError: If the index file or metadata file is not found.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = self.faiss.read_index(index_path)

        metadata_path = f"{index_path}.metadata"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.dimension = self.index.d
        self.embedding_model = self.metadata.get('embedding_model')

    @traced(type="tool")
    def save_index(self, index_path: str) -> None:
        """
        Save the FAISS index and metadata to files.

        Args:
            index_path (str): Path to save the index file.
        """
        self.faiss.write_index(self.index, index_path)
        metadata_path = f"{index_path}.metadata"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    @traced(type="tool")
    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the FAISS index.

        Args:
            vectors (np.ndarray): Array of vectors to add.

        Raises:
            ValueError: If the vector dimension does not match the index dimension.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        if self.metric == "IP":
            # Normalize vectors for Inner Product similarity
            vectors = np.apply_along_axis(self.normalize_vector, 1, vectors)

        self.index.add(vectors)

    @traced(type="tool")
    def search_vectors(self, query_vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the FAISS index.

        Args:
            query_vectors (np.ndarray): Array of query vectors.
            top_k (int, optional): Number of results to return for each query vector. Defaults to 10.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the distances and indices of the top-k results.

        Raises:
            ValueError: If the query vector dimension does not match the index dimension.
        """
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vectors.shape[1]} does not match index dimension {self.dimension}")

        if self.metric == "IP":
            # Normalize query vectors for Inner Product similarity
            query_vectors = np.apply_along_axis(self.normalize_vector, 1, query_vectors)

        distances, indices = self.index.search(query_vectors, top_k)
        return distances, indices

    @traced(type="tool")
    def remove_vectors(self, ids: np.ndarray) -> None:
        """
        Remove vectors from the FAISS index by their IDs.

        Args:
            ids (np.ndarray): Array of vector IDs to remove.
        """
        self.index.remove_ids(ids)

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the FAISS index.

        Returns:
            int: Number of vectors in the index.
        """
        return self.index.ntotal

    @traced(type="tool")
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The normalized vector.
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the index.

        Args:
            key (str): Metadata key.
            value (Any): Metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """
        Get metadata from the index.

        Args:
            key (str): Metadata key.

        Returns:
            Any: Metadata value.
        """
        return self.metadata.get(key)

    def set_embedding_info(self, provider: str, model: str) -> None:
        """
        Set the embedding provider and model information.

        Args:
            provider (str): The embedding provider (e.g., "openai").
            model (str): The embedding model name.
        """
        self.embedding_provider = provider
        self.embedding_model = model
        self.set_metadata('embedding_provider', provider)
        self.set_metadata('embedding_model', model)