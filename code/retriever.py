"""
Hybrid retrieval module combining BM25 (lexical) and vector (semantic) search
with Reciprocal Rank Fusion (RRF).
"""

import hashlib
import re
import numpy as np


class Retriever:
    def __init__(self, indices: dict, embedder):
        """
        indices: dict mapping domain_name -> {'chunks': [...], 'embeddings': np.ndarray, 'bm25': BM25Okapi}
        embedder: object with .encode(texts: list[str]) -> np.ndarray method
        """
        self.indices = indices
        self.embedder = embedder

    def normalize_company(self, company: str | None) -> str | None:
        """Normalize company name to a known domain key."""
        if company is None:
            return None
        company = company.strip()
        if company in ('', 'None', 'none', 'NONE'):
            return None
        lower = company.lower()
        domain_map = {
            'hackerrank': 'hackerrank',
            'claude': 'claude',
            'visa': 'visa',
        }
        return domain_map.get(lower, None)

    def route_domain(self, company: str | None, issue: str) -> str | list[str]:
        """
        Determine which domain(s) to search based on company and issue text.
        Returns a single domain string or a list of domains.
        """
        normalized = self.normalize_company(company)
        if normalized is not None and normalized in self.indices:
            return normalized

        # Infer domain from issue text using keyword matching
        domain_keywords = {
            'hackerrank': ['hackerrank', 'test', 'candidate', 'assessment', 'interview', 'coding', 'screen'],
            'claude': ['claude', 'anthropic', 'ai', 'conversation', 'model', 'bedrock', 'prompt'],
            'visa': ['visa', 'card', 'payment', 'transaction', 'bank', 'merchant', 'cheque', 'travel'],
        }

        issue_lower = issue.lower()
        scores = {}
        for domain in self.indices:
            if domain in domain_keywords:
                keywords = domain_keywords[domain]
            else:
                keywords = []
            count = sum(1 for kw in keywords if kw in issue_lower)
            scores[domain] = count

        if not scores:
            return list(self.indices.keys())

        max_score = max(scores.values())
        if max_score == 0:
            return list(self.indices.keys())

        top_domains = [d for d, s in scores.items() if s == max_score]
        if len(top_domains) == 1:
            return top_domains[0]
        return top_domains

    def bm25_search(self, query: str, domain: str, top_k: int = 20) -> list[tuple[dict, float]]:
        """Lexical search using BM25."""
        if not query or domain not in self.indices:
            return []

        tokenized_query = re.findall(r'\w+', query.lower())
        if not tokenized_query:
            return []

        index = self.indices[domain]
        chunks = index['chunks']
        bm25 = index['bm25']

        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((chunks[idx], score))

        return results

    def vector_search(self, query: str, domain: str, top_k: int = 20, min_similarity: float = 0.15) -> list[tuple[dict, float]]:
        """Semantic search using vector cosine similarity."""
        if not query or domain not in self.indices:
            return []

        index = self.indices[domain]
        embeddings = index['embeddings']
        chunks = index['chunks']

        if embeddings is None or len(embeddings) == 0:
            return []

        # Encode query
        query_vec = self.embedder.encode([query])[0]  # shape (dim,)

        # Cosine similarity via dot product (assumes L2-normalized vectors)
        scores = embeddings @ query_vec  # shape (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(scores[idx])
            if sim >= min_similarity:
                results.append((chunks[idx], sim))

        return results

    def rrf_fusion(self, results_lists: list[list[tuple[dict, float]]], k: int = 60) -> list[tuple[dict, float]]:
        """
        Reciprocal Rank Fusion: combine multiple ranked lists into one.
        RRF score = sum(1 / (k + rank)) for each list where a chunk appears.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for results in results_lists:
            for rank, (chunk, _score) in enumerate(results, start=1):
                # Stable identity via hash of full text + source path
                raw = chunk.get('text', '') + '||' + chunk.get('source_path', '')
                identity = hashlib.md5(raw.encode('utf-8', errors='replace')).hexdigest()
                if identity not in rrf_scores:
                    rrf_scores[identity] = 0.0
                    chunk_map[identity] = chunk
                rrf_scores[identity] += 1.0 / (k + rank)

        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[identity], score) for identity, score in sorted_items]

    def apply_source_priors(self, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """
        Apply source-based score adjustments to boost/penalize certain document types.
        """
        adjusted = []
        for chunk, score in results:
            source_path = chunk.get('source_path', '').lower()
            adjusted_score = score

            if 'index.md' in source_path:
                adjusted_score *= 0.3
            if 'release-notes' in source_path or 'release_notes' in source_path:
                adjusted_score *= 0.5
            if 'integrations/applicant-tracking' in source_path:
                adjusted_score *= 0.7
            if 'faq' in source_path or 'frequently-asked' in source_path or 'troubleshooting' in source_path:
                adjusted_score *= 1.3

            adjusted.append((chunk, adjusted_score))

        # Re-sort by adjusted score descending
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def retrieve(self, query: str, company: str | None, top_k: int = 10) -> list[dict]:
        """
        Main retrieval method: routes domain, performs hybrid search, fuses, and returns top_k chunks.
        """
        if not query:
            return []

        domain = self.route_domain(company, query)

        # Collect result lists from all targeted domains
        result_lists = []

        if isinstance(domain, str):
            domains_to_search = [domain]
        else:
            domains_to_search = domain

        for d in domains_to_search:
            if d not in self.indices:
                continue
            bm25_results = self.bm25_search(query, d, top_k=20)
            vec_results = self.vector_search(query, d, top_k=20)
            if bm25_results:
                result_lists.append(bm25_results)
            if vec_results:
                result_lists.append(vec_results)

        if not result_lists:
            return []

        # Fuse all result lists
        fused = self.rrf_fusion(result_lists)

        # Apply source priors
        adjusted = self.apply_source_priors(fused)

        # Take top_k and add score to each chunk dict
        output = []
        for chunk, score in adjusted[:top_k]:
            result = dict(chunk)  # copy to avoid mutating original
            result['score'] = score
            output.append(result)

        return output
