"""
AutoRAGEvals RAG pipeline.

Config params are at the top of the file; individual functions also accept
keyword arguments to override them at call time.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_COLLECTION_NAME = "autorag_collection"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Imports (deferred so config can be patched before import side-effects)
# ---------------------------------------------------------------------------
from typing import Optional
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import MockLLM
from llama_index.llms.openai import OpenAI


class RAGPipeline:
    """
    LlamaIndex + ChromaDB RAG pipeline.

    Parameters
    ----------
    chunk_size : int
        Token chunk size for the node parser.
    chunk_overlap : int
        Token overlap between consecutive chunks.
    collection_name : str
        ChromaDB collection name.
    chroma_path : str
        Directory where ChromaDB persists data.
    embed_model : str
        OpenAI embedding model name.
    llm_model : str
        OpenAI chat model used for synthesis.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.chroma_path = chroma_path

        # Configure LlamaIndex globals
        Settings.embed_model = OpenAIEmbedding(
            model=embed_model,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        Settings.llm = OpenAI(
            model=llm_model,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # ChromaDB client + collection
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            collection_name
        )

        vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection
        )
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        self._index: Optional[VectorStoreIndex] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_documents(
        self,
        passages: list[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """
        Chunk, embed, and store a list of text passages.

        Parameters
        ----------
        passages : list[str]
            Raw text passages to ingest.
        chunk_size : int, optional
            Override the instance-level chunk size.
        chunk_overlap : int, optional
            Override the instance-level chunk overlap.
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        documents = [Document(text=p) for p in passages]

        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            transformations=[node_parser],
            show_progress=True,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict:
        """
        Run a RAG query and return the synthesised answer plus source nodes.

        Parameters
        ----------
        question : str
            The question to answer.
        top_k : int
            Number of context chunks to retrieve.

        Returns
        -------
        dict with keys:
            - ``answer``  : str — synthesised answer from the LLM
            - ``contexts``: list[str] — retrieved passage chunks
        """
        if self._index is None:
            # Reconstruct index from the persisted ChromaDB collection
            vector_store = ChromaVectorStore(
                chroma_collection=self._chroma_collection
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )

        query_engine = self._index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)

        contexts = [
            node.get_content() for node in response.source_nodes
        ]

        return {
            "answer": str(response),
            "contexts": contexts,
        }
