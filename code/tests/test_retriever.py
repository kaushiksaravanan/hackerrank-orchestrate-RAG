"""Tests retriever with REAL indices. Verifies correct articles are found for known queries."""
import numpy as np
import pytest


def test_normalize_company(retriever_instance):
    """Test company name normalization."""
    assert retriever_instance.normalize_company('HackerRank') == 'hackerrank'
    assert retriever_instance.normalize_company('Visa') == 'visa'
    assert retriever_instance.normalize_company('Claude') == 'claude'
    assert retriever_instance.normalize_company('None ') is None
    assert retriever_instance.normalize_company('') is None
    assert retriever_instance.normalize_company(None) is None
    assert retriever_instance.normalize_company('none') is None


def test_bm25_pause_subscription(retriever_instance):
    """BM25 for 'pause subscription' must find relevant HackerRank chunk."""
    results = retriever_instance.bm25_search('pause subscription', 'hackerrank', 10)
    assert len(results) >= 1, "No BM25 results for 'pause subscription'"
    top_text = results[0][0]['text'].lower()
    assert 'pause' in top_text or 'subscription' in top_text, \
        f"Top result irrelevant: {top_text[:100]}"


def test_bm25_claudebot(retriever_instance):
    """BM25 for 'claudebot robots.txt crawl' must find the crawling article."""
    results = retriever_instance.bm25_search('claudebot robots.txt crawl', 'claude', 5)
    assert len(results) >= 1
    all_text = ' '.join(r[0]['text'] for r in results)
    assert 'claudebot@anthropic.com' in all_text or 'claudebot' in all_text.lower(), \
        "ClaudeBot article not found in BM25 results"


def test_vector_search_visa_minimum(retriever_instance):
    """Vector search for 'minimum transaction amount' must find US$10 content."""
    results = retriever_instance.vector_search('minimum transaction amount visa card ten dollars', 'visa', 10)
    assert len(results) >= 1
    all_text = ' '.join(r[0]['text'] for r in results[:5])
    assert 'US$10' in all_text or 'minimum' in all_text.lower() or '$10' in all_text, \
        f"US$10 content not found in vector results"


def test_retrieve_full_pipeline(retriever_instance):
    """Full retrieve for 'dispute a charge' with Visa must return relevant chunks."""
    results = retriever_instance.retrieve('how to dispute a charge on my visa card', 'Visa', 5)
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    # Check structure
    for chunk in results:
        assert 'text' in chunk
        assert 'domain' in chunk
        assert 'score' in chunk
    # At least one must be from visa domain
    domains = [c['domain'] for c in results]
    assert 'visa' in domains, f"No visa chunks in results for Visa query: {domains}"
    # At least one must mention dispute
    texts = ' '.join(c['text'].lower() for c in results)
    assert 'dispute' in texts or 'charge' in texts, "No dispute content in results"


def test_domain_inference_visa(retriever_instance):
    """route_domain with None company + Visa keywords must infer 'visa'."""
    domain = retriever_instance.route_domain(None, 'my Visa card was stolen during international travel')
    # Should return 'visa' or a list containing 'visa'
    if isinstance(domain, list):
        assert 'visa' in domain, f"Visa not in inferred domains: {domain}"
    else:
        assert domain == 'visa', f"Expected 'visa', got '{domain}'"


def test_domain_inference_hackerrank(retriever_instance):
    """route_domain with None company + HackerRank keywords must infer 'hackerrank'."""
    domain = retriever_instance.route_domain(None, 'my HackerRank test score is wrong and the candidate assessment failed')
    if isinstance(domain, list):
        assert 'hackerrank' in domain
    else:
        assert domain == 'hackerrank', f"Expected 'hackerrank', got '{domain}'"


def test_rrf_scores_descending(retriever_instance):
    """RRF fusion output must be sorted by descending score."""
    results = retriever_instance.retrieve('lost card emergency', 'Visa', 10)
    scores = [r['score'] for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Scores not descending at position {i}: {scores[i]} < {scores[i+1]}"
