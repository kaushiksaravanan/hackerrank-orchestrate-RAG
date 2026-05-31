"""Tests that embeddings are REAL neural embeddings, not random/fake.
Key anti-fake mechanism: semantic similarity ordering.
Random vectors give cosine ~0 for ALL pairs. Real embeddings give high sim for related texts."""
import numpy as np
import pytest


def test_semantic_similarity_ordering(embedder):
    """Related texts must be more similar than unrelated texts. FAILS with random vectors."""
    v1 = embedder.encode(['pause subscription billing'])[0]
    v2 = embedder.encode(['cancel my plan payments'])[0]
    v3 = embedder.encode(['iron man actor movie name'])[0]
    sim_related = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v3)
    assert sim_related > 0.4, f"Related texts too dissimilar: {sim_related:.3f}"
    assert sim_unrelated < sim_related, f"Unrelated texts more similar than related: {sim_unrelated:.3f} >= {sim_related:.3f}"


def test_visa_semantic_similarity(embedder):
    """Visa-related queries must cluster together."""
    v1 = embedder.encode(['lost visa card india report stolen'])[0]
    v2 = embedder.encode(['report stolen credit card emergency call bank'])[0]
    v3 = embedder.encode(['python programming tutorial for beginners'])[0]
    sim_related = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v3)
    assert sim_related > 0.35, f"Visa queries too dissimilar: {sim_related:.3f}"
    assert sim_unrelated < sim_related


def test_l2_normalization(embedder):
    """All embeddings must be L2-normalized (norm == 1.0). FAILS with zeros or unnormalized."""
    vecs = embedder.encode(['hello world', 'test query', 'another sentence'])
    norms = np.linalg.norm(vecs, axis=1)
    for i, norm in enumerate(norms):
        assert abs(norm - 1.0) < 0.01, f"Vector {i} not normalized: norm={norm:.4f}"


def test_determinism(embedder):
    """Same input must produce identical output. FAILS with random generation."""
    v1 = embedder.encode(['determinism test string'])[0]
    v2 = embedder.encode(['determinism test string'])[0]
    assert np.allclose(v1, v2, atol=1e-6), "Non-deterministic embeddings"


def test_batch_shape(embedder):
    """Batch encoding must return correct shape."""
    texts = ['one', 'two', 'three', 'four', 'five']
    vecs = embedder.encode(texts)
    assert vecs.shape[0] == 5, f"Wrong batch size: {vecs.shape[0]}"
    assert vecs.shape[1] >= 100, f"Embedding dim too small: {vecs.shape[1]}"


def test_different_texts_different_vectors(embedder):
    """Different texts must produce different vectors. FAILS if encode returns constant."""
    v1 = embedder.encode(['hello world'])[0]
    v2 = embedder.encode(['goodbye universe'])[0]
    # Cosine similarity (dot product of L2-normalized vectors) must be < 0.99
    cosine_sim = float(np.dot(v1, v2))
    assert cosine_sim < 0.99, f"Different texts produce near-identical vectors (cosine={cosine_sim:.4f})"
    assert not np.array_equal(v1, v2), "Different texts produce identical vectors"
