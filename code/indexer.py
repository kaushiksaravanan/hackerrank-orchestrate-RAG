"""
Indexing module for the support triage agent.

Loads corpus documents, chunks them by heading, builds BM25 + vector indices,
and caches them to disk for fast reuse across sessions.
"""

import hashlib
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

from config import CACHE_DIR, CHUNK_OVERLAP, DATA_DIR, DOMAINS, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, SKIP_FILES
from preprocessor import clean_document_full, detect_stub

logger = logging.getLogger(__name__)

# Cache the tiktoken encoder at module level (expensive to create per call)
_ENCODER = tiktoken.encoding_for_model("gpt-4")


# ─── 1. Load Documents ───────────────────────────────────────────────────────


def load_documents(domain: str) -> list[dict]:
    """
    Load all markdown documents for a given domain.

    Globs all .md files recursively under DATA_DIR / domain, skips files in
    SKIP_FILES, skips index.md (table of contents), and skips stub documents.

    Args:
        domain: One of 'hackerrank', 'claude', 'visa'.

    Returns:
        List of document dicts with keys: path, content, domain, title, metadata.
    """
    domain_path = DATA_DIR / domain
    if not domain_path.exists():
        logger.warning("Domain path does not exist: %s", domain_path)
        return []

    documents = []

    for md_file in sorted(domain_path.glob("**/*.md")):
        # Skip files in SKIP_FILES set
        if md_file.name in SKIP_FILES:
            continue

        # Skip index.md files (tables of contents)
        if md_file.name == "index.md":
            continue

        # Read file content
        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to read %s: %s", md_file, e)
            continue

        # Skip stub documents
        if detect_stub(content):
            continue

        # Clean the document (full pipeline handles Cloudflare email artifacts)
        metadata, body = clean_document_full(content, domain)

        if not body.strip():
            continue

        documents.append({
            "path": str(md_file.relative_to(DATA_DIR)),
            "content": body,
            "domain": domain,
            "title": metadata.get("clean_title", ""),
            "metadata": metadata,
        })

    logger.info("Loaded %d documents for domain '%s'", len(documents), domain)
    return documents


# ─── 2. Split by Headings ────────────────────────────────────────────────────


def split_by_headings(body: str) -> list[dict]:
    """
    Split document body into sections based on markdown headings (h1-h4).

    Keeps tables and numbered lists atomic within their sections.

    Args:
        body: Cleaned markdown body text.

    Returns:
        List of section dicts with keys: heading, level, text.
    """
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    sections = []
    matches = list(heading_pattern.finditer(body))

    if not matches:
        # No headings at all — entire body is one section
        return [{"heading": "", "level": 0, "text": body.strip()}]

    # Content before the first heading
    pre_heading_text = body[: matches[0].start()].strip()
    if pre_heading_text:
        sections.append({"heading": "", "level": 0, "text": pre_heading_text})

    # Each heading starts a section that runs until the next heading
    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        heading_level = len(match.group(1))

        # Text runs from end of this heading line to start of next heading (or end)
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[text_start:text_end].strip()

        sections.append({
            "heading": heading_text,
            "level": heading_level,
            "text": section_text,
        })

    return sections


# ─── 3. Chunk Sections ───────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Count tokens using the cached tiktoken encoder."""
    return len(_ENCODER.encode(text))


def _is_table_line(line: str) -> bool:
    """Check if a line is part of a markdown table (contains | delimiters)."""
    stripped = line.strip()
    return "|" in stripped and len(stripped) > 1


def _is_numbered_list_item(line: str) -> bool:
    """Check if a line starts a numbered list item."""
    return bool(re.match(r"^\s*\d+[.)]\s+", line))


def _split_at_paragraph_boundaries(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Split text at paragraph boundaries (\n\n) respecting max_tokens.
    Adds overlap tokens from end of previous chunk to start of next.
    Keeps tables and numbered lists atomic.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # Check if this paragraph is part of a table or numbered list
        # that shouldn't be split
        lines = para.split("\n")
        is_table = any(_is_table_line(line) for line in lines)
        is_numbered_list = all(
            _is_numbered_list_item(line) or line.strip() == ""
            for line in lines
            if line.strip()
        )

        if current_tokens + para_tokens <= max_tokens:
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            # Current chunk is full — save it
            if current_parts:
                chunks.append("\n\n".join(current_parts))

            # If a single paragraph exceeds max_tokens and it's a table/list,
            # keep it as one chunk anyway (don't split mid-table)
            if para_tokens > max_tokens and (is_table or is_numbered_list):
                chunks.append(para)
                current_parts = []
                current_tokens = 0
            elif para_tokens > max_tokens:
                # Large paragraph that's not a table — split by sentences/lines
                # as a fallback, but generally just keep it
                chunks.append(para)
                current_parts = []
                current_tokens = 0
            else:
                current_parts = [para]
                current_tokens = para_tokens

    # Don't forget the last chunk
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Apply overlap: prepend tokens from end of previous chunk to start of next
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # Get overlap text from end of previous chunk
            prev_tokens = _ENCODER.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
            overlap_text = _ENCODER.decode(overlap_tokens)
            overlapped_chunks.append(overlap_text.strip() + "\n\n" + chunks[i])
        return overlapped_chunks

    return chunks


def chunk_sections(
    sections: list[dict],
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Convert sections into appropriately-sized chunks.

    - Sections smaller than min_tokens are merged with the next section.
    - Sections within max_tokens are kept as one chunk.
    - Sections exceeding max_tokens are split at paragraph boundaries with overlap.

    Args:
        sections: List of section dicts from split_by_headings().
        max_tokens: Maximum tokens per chunk.
        min_tokens: Minimum tokens per chunk (merge if below).
        overlap: Number of overlap tokens between consecutive sub-chunks.

    Returns:
        List of chunk dicts with keys: heading, text, tokens.
    """
    chunks = []

    # First pass: merge small sections with next
    merged_sections = []
    i = 0
    while i < len(sections):
        section = sections[i]
        tokens = _count_tokens(section["text"])

        if tokens < min_tokens and i + 1 < len(sections):
            # Merge with next section
            next_section = sections[i + 1]
            merged_text = section["text"] + "\n\n" + next_section["text"]
            merged_heading = next_section["heading"] if next_section["heading"] else section["heading"]
            merged_level = next_section["level"] if next_section["level"] else section["level"]
            merged_sections.append({
                "heading": merged_heading,
                "level": merged_level,
                "text": merged_text,
            })
            i += 2
        else:
            merged_sections.append(section)
            i += 1

    # Second pass: chunk each merged section
    for section in merged_sections:
        text = section["text"]
        tokens = _count_tokens(text)

        if not text.strip():
            continue

        if tokens <= max_tokens:
            # Fits in one chunk
            chunks.append({
                "heading": section["heading"],
                "text": text,
                "tokens": tokens,
            })
        else:
            # Need to split at paragraph boundaries
            sub_chunks = _split_at_paragraph_boundaries(text, max_tokens, overlap)
            for sub_text in sub_chunks:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue
                sub_tokens = _count_tokens(sub_text)
                if sub_tokens < min_tokens:
                    continue
                chunks.append({
                    "heading": section["heading"],
                    "text": sub_text,
                    "tokens": sub_tokens,
                })

    return chunks


# ─── 4. Add Contextual Headers ──────────────────────────────────────────────


def add_contextual_headers(chunks: list[dict], doc_metadata: dict) -> list[dict]:
    """
    Prepend contextual information to each chunk's text for better retrieval.

    Adds domain, article title, and section heading as a prefix.

    Args:
        chunks: List of chunk dicts from chunk_sections().
        doc_metadata: Document metadata dict with 'domain' and 'title' keys.

    Returns:
        Modified chunks with contextual headers prepended to text.
    """
    domain = doc_metadata.get("domain", "")
    title = doc_metadata.get("title", "")

    for chunk in chunks:
        heading = chunk.get("heading", "")
        header = f"[Domain: {domain}] [Article: {title}] [Section: {heading}]\n"
        chunk["text"] = header + chunk["text"]
        # Update token count
        chunk["tokens"] = _count_tokens(chunk["text"])

    return chunks


# ─── 5. Tokenize for BM25 ───────────────────────────────────────────────────


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenization for BM25: lowercase and extract word tokens.

    Args:
        text: Input text.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\w+", text.lower())


# ─── 6. Build Domain Index ───────────────────────────────────────────────────


def build_domain_index(domain: str, embedder) -> dict:
    """
    Build a complete retrieval index for a single domain.

    Loads documents, splits into chunks, builds BM25 and vector indices.

    Args:
        domain: One of 'hackerrank', 'claude', 'visa'.
        embedder: An embedder instance with an encode(texts) method.

    Returns:
        Dict with keys: chunks, embeddings, bm25.
    """
    documents = load_documents(domain)
    all_chunks = []

    for doc in documents:
        # Split into sections by heading
        sections = split_by_headings(doc["content"])

        # Chunk the sections
        chunks = chunk_sections(sections)

        # Add contextual headers
        doc_meta = {
            "domain": doc["domain"],
            "title": doc["title"],
        }
        chunks = add_contextual_headers(chunks, doc_meta)

        # Enrich each chunk with document-level metadata
        for chunk in chunks:
            chunk["domain"] = doc["domain"]
            chunk["title"] = doc["title"]
            chunk["section"] = chunk["heading"]
            chunk["source_path"] = doc["path"]

        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced for domain '%s'", domain)
        return {"chunks": [], "embeddings": np.array([]), "bm25": None}

    print(f"Building index for {domain}... {len(all_chunks)} chunks")

    # Build BM25 index
    tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Build vector embeddings
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(chunk_texts)

    return {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "bm25": bm25,
    }


# ─── 7. Build All Indices ────────────────────────────────────────────────────


def build_all_indices(embedder, force_rebuild: bool = False) -> dict[str, dict]:
    """
    Build or load cached indices for all configured domains.

    Args:
        embedder: An embedder instance with an encode(texts) method.
        force_rebuild: If True, rebuild indices even if cache exists.

    Returns:
        Dict mapping domain name to its index dict (chunks, embeddings, bm25).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    indices = {}

    # Model-aware cache key: hash the model name so indices auto-invalidate
    # when the embedding model changes.
    model_name = getattr(embedder, "model_name", "unknown")
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]

    for domain in DOMAINS:
        cache_path = CACHE_DIR / f"{domain}_{model_hash}_index.pkl"

        if cache_path.exists() and not force_rebuild:
            print(f"Loading cached index for {domain}...")
            try:
                with open(cache_path, "rb") as f:
                    indices[domain] = pickle.load(f)
                print(f"  Loaded {len(indices[domain]['chunks'])} chunks from cache")
                continue
            except (pickle.UnpicklingError, EOFError, KeyError) as e:
                logger.warning("Cache corrupted for %s, rebuilding: %s", domain, e)

        # Build fresh index
        index = build_domain_index(domain, embedder)
        indices[domain] = index

        # Save to cache with protocol 5 for speed
        if index["chunks"]:
            with open(cache_path, "wb") as f:
                pickle.dump(index, f, protocol=5)
            print(f"  Cached index for {domain} ({len(index['chunks'])} chunks)")

    return indices


# ─── CLI Entry Point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from embeddings import create_embedder

    print("Initializing embedder...")
    embedder = create_embedder()

    force = "--force" in sys.argv
    indices = build_all_indices(embedder, force_rebuild=force)

    total_chunks = sum(len(idx["chunks"]) for idx in indices.values())
    print(f"\nDone. Total chunks across all domains: {total_chunks}")

    for domain, idx in indices.items():
        if idx["chunks"]:
            avg_tokens = sum(c["tokens"] for c in idx["chunks"]) / len(idx["chunks"])
            print(f"  {domain}: {len(idx['chunks'])} chunks, avg {avg_tokens:.0f} tokens/chunk")
