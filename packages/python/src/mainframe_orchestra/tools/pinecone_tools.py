# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import List, Dict, Any
import numpy as np

def check_pinecone():
    try:
        import pinecone
    except ModuleNotFoundError:
        raise ImportError("pinecone is required for Pinecone tools. Install with `pip install pinecone`")
    return pinecone

class PineconeTools:
    def __init__(self, api_key: str = None):
        """
        Initialize PineconeTools with the Pinecone API key.

        Args:
            api_key (str, optional): Pinecone API key. If not provided, it will try to use the PINECONE_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Please provide it or set the PINECONE_API_KEY environment variable.")
        self.pc = check_pinecone().Pinecone(api_key=self.api_key)

    def get_pinecone_index(self, name: str):
        pinecone_client = check_pinecone().Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return pinecone_client.Index(name)

    def create_index(self, name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1") -> None:
        """
        Create a new Pinecone index.

        Args:
            name (str): Name of the index to create.
            dimension (int): Dimension of the vectors to be stored in the index.
            metric (str, optional): Distance metric to use. Defaults to "cosine".
            cloud (str, optional): Cloud provider. Defaults to "aws".
            region (str, optional): Cloud region. Defaults to "us-east-1".

        Raises:
            Exception: If there's an error creating the index.
        """
        try:
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=check_pinecone().ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Index '{name}' created successfully.")
        except Exception as e:
            raise Exception(f"Error creating index: {str(e)}")

    def delete_index(self, name: str) -> None:
        """
        Delete a Pinecone index.

        Args:
            name (str): Name of the index to delete.

        Raises:
            Exception: If there's an error deleting the index.
        """
        try:
            self.pc.delete_index(name)
            print(f"Index '{name}' deleted successfully.")
        except Exception as e:
            raise Exception(f"Error deleting index: {str(e)}")

    def list_indexes(self) -> List[str]:
        """
        List all available Pinecone indexes.

        Returns:
            List[str]: List of index names.

        Raises:
            Exception: If there's an error listing the indexes.
        """
        try:
            return self.pc.list_indexes()
        except Exception as e:
            raise Exception(f"Error listing indexes: {str(e)}")

    def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> None:
        """
        Upsert vectors into a Pinecone index.

        Args:
            index_name (str): Name of the index to upsert vectors into.
            vectors (List[Dict[str, Any]]): List of vectors to upsert. Each vector should be a dictionary with 'id', 'values', and optionally 'metadata'.

        Raises:
            Exception: If there's an error upserting vectors.
        """
        try:
            index = self.pc.Index(index_name)
            index.upsert(vectors=vectors)
            print(f"Vectors upserted successfully into index '{index_name}'.")
        except Exception as e:
            raise Exception(f"Error upserting vectors: {str(e)}")

    def query_index(self, index_name: str, query_vector: List[float], top_k: int = 10, filter: Dict = None, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query a Pinecone index for similar vectors.

        Args:
            index_name (str): Name of the index to query.
            query_vector (List[float]): The query vector.
            top_k (int, optional): Number of results to return. Defaults to 10.
            filter (Dict, optional): Metadata filter to apply to the query. Defaults to None.
            include_metadata (bool, optional): Whether to include metadata in the results. Defaults to True.

        Returns:
            Dict[str, Any]: Query results containing matches and their scores.

        Raises:
            Exception: If there's an error querying the index.
        """
        try:
            index = self.pc.Index(index_name)
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter
            )
            return results
        except Exception as e:
            raise Exception(f"Error querying index: {str(e)}")

    def delete_vectors(self, index_name: str, ids: List[str]) -> None:
        """
        Delete vectors from a Pinecone index by their IDs.

        Args:
            index_name (str): Name of the index to delete vectors from.
            ids (List[str]): List of vector IDs to delete.

        Raises:
            Exception: If there's an error deleting vectors.
        """
        try:
            index = self.pc.Index(index_name)
            index.delete(ids=ids)
            print(f"Vectors deleted successfully from index '{index_name}'.")
        except Exception as e:
            raise Exception(f"Error deleting vectors: {str(e)}")

    def update_vector_metadata(self, index_name: str, id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the metadata of a vector in a Pinecone index.

        Args:
            index_name (str): Name of the index containing the vector.
            id (str): ID of the vector to update.
            metadata (Dict[str, Any]): New metadata to assign to the vector.

        Raises:
            Exception: If there's an error updating the vector metadata.
        """
        try:
            index = self.pc.Index(index_name)
            index.update(id=id, set_metadata=metadata)
            print(f"Metadata updated successfully for vector '{id}' in index '{index_name}'.")
        except Exception as e:
            raise Exception(f"Error updating vector metadata: {str(e)}")

    def describe_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about a Pinecone index.

        Args:
            index_name (str): Name of the index to describe.

        Returns:
            Dict[str, Any]: Statistics about the index.

        Raises:
            Exception: If there's an error describing the index stats.
        """
        try:
            index = self.pc.Index(index_name)
            return index.describe_index_stats()
        except Exception as e:
            raise Exception(f"Error describing index stats: {str(e)}")

    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector (List[float]): The input vector.

        Returns:
            List[float]: The normalized vector.
        """
        norm = np.linalg.norm(vector)
        return (np.array(vector) / norm).tolist() if norm != 0 else vector