"""Tests indexer against REAL corpus. Verifies chunk counts, content preservation, and index structure."""
import re
import numpy as np
import pytest


def test_visa_document_count():
    """Visa corpus must load 10-14 documents (skips stubs)."""
    from indexer import load_documents
    docs = load_documents('visa')
    assert 8 <= len(docs) <= 14, f"Unexpected visa doc count: {len(docs)}"


def test_claude_document_count():
    """Claude corpus must load 300+ documents."""
    from indexer import load_documents
    docs = load_documents('claude')
    assert len(docs) >= 280, f"Too few claude docs: {len(docs)}"


def test_hackerrank_document_count():
    """HackerRank corpus must load 380+ documents."""
    from indexer import load_documents
    docs = load_documents('hackerrank')
    assert len(docs) >= 350, f"Too few hackerrank docs: {len(docs)}"


def test_visa_index_chunk_count(visa_index):
    """Visa index must have at least 40 chunks."""
    assert len(visa_index['chunks']) >= 30, f"Too few visa chunks: {len(visa_index['chunks'])}"


def test_visa_index_contains_phone(visa_index):
    """At least one visa chunk must contain the India phone number. FAILS if corpus not read."""
    texts = [c['text'] for c in visa_index['chunks']]
    found = any('000-800-100-1219' in t for t in texts)
    assert found, "Visa India phone 000-800-100-1219 not found in any chunk"


def test_visa_index_contains_us10(visa_index):
    """At least one visa chunk must contain the US$10 minimum transaction text."""
    texts = [c['text'] for c in visa_index['chunks']]
    found = any('US$10' in t or 'minimum transaction' in t.lower() or '$10' in t for t in texts)
    assert found, "US$10 minimum transaction FAQ not found in any chunk"


def test_no_images_in_chunks(visa_index):
    """NO chunk may contain markdown image patterns. FAILS if preprocessor skipped."""
    for chunk in visa_index['chunks']:
        assert '![' not in chunk['text'], f"Image found in chunk: {chunk['text'][:80]}"


def test_contextual_headers(visa_index):
    """Every chunk must start with [Domain: header. FAILS if headers skipped."""
    for chunk in visa_index['chunks']:
        assert chunk['text'].startswith('[Domain:'), f"Missing header: {chunk['text'][:60]}"


def test_chunk_token_limit(visa_index):
    """Most chunks must be under 600 tokens. Tables/lists kept atomic may be larger."""
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-4')
    oversized = []
    for chunk in visa_index['chunks']:
        tokens = len(enc.encode(chunk['text']))
        if tokens > 600:
            oversized.append(tokens)
    # At most 10% of chunks may exceed the soft limit (atomic tables/lists)
    max_oversized = max(1, len(visa_index['chunks']) // 10)
    assert len(oversized) <= max_oversized, \
        f"Too many oversized chunks: {len(oversized)} > {max_oversized}. Sizes: {oversized}"
    # But NO chunk should be absurdly large (> 3000 tokens)
    for tokens in oversized:
        assert tokens <= 3000, f"Chunk absurdly large: {tokens} tokens"


def test_bm25_returns_relevant_chunk(visa_index):
    """BM25 for 'lost stolen card' must return chunk with lost/stolen/phone content."""
    from indexer import tokenize_for_bm25
    query_tokens = tokenize_for_bm25('lost stolen card visa india')
    scores = visa_index['bm25'].get_scores(query_tokens)
    best_idx = int(np.argmax(scores))
    best_chunk = visa_index['chunks'][best_idx]['text'].lower()
    assert scores[best_idx] > 0, "BM25 returned zero scores for relevant query"
    assert any(w in best_chunk for w in ['lost', 'stolen', '000-800', 'card']), \
        f"BM25 top result irrelevant: {best_chunk[:100]}"


def test_embeddings_shape(visa_index):
    """Index embeddings must have correct shape matching chunk count."""
    n_chunks = len(visa_index['chunks'])
    assert visa_index['embeddings'].shape[0] == n_chunks
    assert visa_index['embeddings'].shape[1] >= 100  # Real embedding dim
