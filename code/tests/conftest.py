"""
Shared pytest fixtures for the support triage agent test suite.

Session-scoped fixtures for expensive operations (embedder, indices).
"""

import sys
from pathlib import Path

# Add code/ directory to sys.path so module imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def repo_root():
    """Return the repository root path."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def data_dir():
    """Return DATA_DIR from config."""
    from config import DATA_DIR
    return DATA_DIR


@pytest.fixture(scope="session")
def embedder():
    """
    Create an embedder instance shared across all tests.
    Tries ONNXEmbedder first, falls back to TFIDFEmbedder.
    """
    from embeddings import create_embedder
    return create_embedder()


@pytest.fixture(scope="session")
def visa_index(embedder):
    """Build visa domain index only (smallest, fastest for unit tests)."""
    from indexer import build_domain_index
    return build_domain_index("visa", embedder)


@pytest.fixture(scope="session")
def all_indices(embedder):
    """Build all domain indices (hackerrank, claude, visa)."""
    from indexer import build_all_indices
    return build_all_indices(embedder, force_rebuild=False)


@pytest.fixture(scope="session")
def retriever_instance(all_indices, embedder):
    """Create a Retriever instance with all indices."""
    from retriever import Retriever
    return Retriever(all_indices, embedder)


@pytest.fixture(scope="session")
def llm_client():
    """Create an LLMClient instance shared across all tests."""
    from llm import create_llm_client
    return create_llm_client()
