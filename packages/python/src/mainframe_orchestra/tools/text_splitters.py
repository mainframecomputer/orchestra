# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from typing import List, Union
import numpy as np
from dotenv import load_dotenv
from .embedding_tools import EmbeddingsTools
import igraph as ig
import leidenalg as la
from sentence_splitter import SentenceSplitter as ExternalSentenceSplitter
load_dotenv()

class SemanticSplitter:
    def __init__(self, embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small"):
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

    @staticmethod
    def chunk_text(text: Union[str, List[str]], rearrange: bool = False, 
                   embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small") -> List[str]:
        splitter = SemanticSplitter(embedding_provider, embedding_model)
        
        if isinstance(text, str):
            return splitter._process_single_text(text, rearrange)
        elif isinstance(text, list):
            all_chunks = []
            for doc in text:
                all_chunks.extend(splitter._process_single_text(doc, rearrange))
            return all_chunks
        else:
            raise ValueError("Input must be either a string or a list of strings")

    def _process_single_text(self, text: str, rearrange: bool) -> List[str]:
        segments = self._create_sentence_segments(text)
        embeddings = self._embed_segments(segments)
        communities = self._detect_communities(embeddings)
        chunks = self._create_chunks_from_communities(segments, communities, rearrange)
        
        print(f"Created {len(chunks)} non-empty chunks for this document")
        return chunks

    def _create_sentence_segments(self, text: str) -> List[str]:
        sentences = SentenceSplitter.split_text_by_sentences(text)
        segments = [sentence.strip() for sentence in sentences]
        print(f"Created {len(segments)} segments")
        return segments

    def _embed_segments(self, segments: List[str]) -> np.ndarray:
        embeddings, _ = EmbeddingsTools.get_embeddings(segments, self.embedding_provider, self.embedding_model)
        return np.array(embeddings)

    def _detect_communities(self, embeddings: np.ndarray) -> List[int]:
        if embeddings.shape[0] < 2:
            return [0]
        
        G = self._create_similarity_graph(embeddings, similarity_threshold=0.55)
        
        partition = self._find_optimal_partition(G, resolution=0.35)
        
        communities = partition.membership
        
        num_communities = len(set(communities))
        print(f"Communities: {num_communities}")
        
        return communities

    def _create_chunks_from_communities(self, segments: List[str], communities: List[int], rearrange: bool) -> List[str]:
        if rearrange:
            # Group segments by community
            community_groups = {}
            for segment, community in zip(segments, communities):
                if community not in community_groups:
                    community_groups[community] = []
                community_groups[community].append(segment)
            
            # Create chunks from rearranged communities
            chunks = [' '.join(group).strip() for group in community_groups.values() if group]
        else:
            # Create chunks respecting original order
            chunks = []
            current_community = communities[0]
            current_chunk = []
            
            for segment, community in zip(segments, communities):
                if community != current_community:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = []
                    current_community = community
                current_chunk.append(segment)
            
            # Add the last chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
        
        return [chunk for chunk in chunks if chunk]  # Remove any empty chunks

    def _identify_breakpoints(self, communities: List[int]) -> List[int]:
        breakpoints = []
        for i in range(1, len(communities)):
            if communities[i] != communities[i-1]:
                breakpoints.append(i)
        return breakpoints

    def _create_similarity_graph(self, embeddings: np.ndarray, similarity_threshold: float) -> ig.Graph:
        similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarities, 0)
        similarities = np.maximum(similarities, 0)
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
        
        # Apply similarity threshold
        adjacency_matrix = (similarities >= similarity_threshold).astype(int)
        
        G = ig.Graph.Adjacency(adjacency_matrix.tolist())
        G.es['weight'] = similarities[np.where(adjacency_matrix)]
        return G

    def _find_optimal_partition(self, G: ig.Graph, resolution: float) -> la.VertexPartition:
        return la.find_partition(
            G, 
            la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution
        )

    def _split_oversized_communities(self, membership: List[int], max_size: int) -> List[int]:
        community_sizes = {}
        for comm in membership:
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        
        new_membership = []
        current_comm = max(membership) + 1
        for i, comm in enumerate(membership):
            if community_sizes[comm] > max_size:
                if i % max_size == 0:
                    current_comm += 1
                new_membership.append(current_comm)
            else:
                new_membership.append(comm)
        
        return new_membership

class SentenceSplitter:
    @staticmethod
    def split_text_by_sentences(text: str, chunk_size: int = 5, overlap: int = 1, language: str = 'en') -> List[str]:
        """
        Split the text into chunks of sentences with overlap.
        
        :param text: The input text to split.
        :param chunk_size: The number of sentences per chunk.
        :param overlap: The number of sentences to overlap between chunks.
        :param language: The language of the text (default: 'en').
        :return: A list of text chunks.
        """
        splitter = ExternalSentenceSplitter(language=language)
        sentences = splitter.split(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk.strip())
        
        print(f"Created {len(chunks)} chunks with {chunk_size} sentences each and {overlap} sentence overlap")
        return chunks
