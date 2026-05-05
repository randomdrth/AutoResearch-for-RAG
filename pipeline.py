"""
AutoRAGEvals RAG pipeline.

Config params are at the top of the file; individual functions also accept
keyword arguments to override them at call time.
"""

import os
from typing import List, Optional
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
# Prompt templates for generation
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "baseline": (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    ),
    "concise": (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query in 1-2 sentences maximum, directly and "
        "without elaboration.\n"
        "Query: {query_str}\n"
        "Answer: "
    ),
    "chain_of_thought": (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "briefly reason step by step before giving the final answer.\n"
        "Query: {query_str}\n"
        "Answer: "
    ),
}

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
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
        passages: List[str],
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
    # Reranker
    # ------------------------------------------------------------------

    def _build_reranker(self, top_n: int):
        """
        Return a postprocessor reranker.

        Uses CohereRerank when COHERE_API_KEY is set and the package is
        installed; falls back to LLMRerank (uses the global Settings.llm).
        """
        cohere_key = os.environ.get("COHERE_API_KEY")
        if cohere_key:
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                return CohereRerank(api_key=cohere_key, top_n=top_n)
            except ImportError:
                pass  # fall through to LLMRerank

        from llama_index.core.postprocessor import LLMRerank
        return LLMRerank(top_n=top_n, choice_batch_size=5)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        prompt_variant: str = "baseline",
        use_reranker: bool = False,
    ) -> dict:
        """
        Run a RAG query and return the synthesised answer plus source nodes.

        Parameters
        ----------
        question : str
            The question to answer.
        top_k : int
            Number of context chunks to retrieve (and keep after reranking).
        prompt_variant : str
            One of "baseline", "concise", "chain_of_thought".
        use_reranker : bool
            If True, retrieve top_k*2 candidates and rerank to top_k.

        Returns
        -------
        dict with keys:
            - ``answer``  : str — synthesised answer from the LLM
            - ``contexts``: list[str] — retrieved passage chunks
        """
        if self._index is None:
            vector_store = ChromaVectorStore(
                chroma_collection=self._chroma_collection
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )

        template_str = PROMPT_TEMPLATES.get(
            prompt_variant, PROMPT_TEMPLATES["baseline"]
        )
        qa_template = PromptTemplate(template_str)

        if use_reranker:
            reranker = self._build_reranker(top_n=top_k)
            rerank_engine = self._index.as_query_engine(
                similarity_top_k=top_k * 2,
                node_postprocessors=[reranker],
                text_qa_template=qa_template,
            )
            try:
                response = rerank_engine.query(question)
            except (ValueError, IndexError):
                # LLMRerank occasionally fails to parse the LLM's ranking
                # output (e.g. returns passage text instead of a number).
                # Fall back to standard retrieval so the run doesn't crash.
                fallback_engine = self._index.as_query_engine(
                    similarity_top_k=top_k,
                    text_qa_template=qa_template,
                )
                response = fallback_engine.query(question)
        else:
            query_engine = self._index.as_query_engine(
                similarity_top_k=top_k,
                text_qa_template=qa_template,
            )
            response = query_engine.query(question)

        contexts = [node.get_content() for node in response.source_nodes]

        return {
            "answer": str(response),
            "contexts": contexts,
        }
