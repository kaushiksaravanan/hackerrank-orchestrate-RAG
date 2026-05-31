"""
Text preprocessing module for the support corpus.

Cleans markdown articles from 3 domains (hackerrank, claude, visa),
each with different formatting quirks. No external dependencies beyond stdlib.
"""

import re
import os
from pathlib import Path


def _parse_yaml_block(yaml_text: str) -> dict:
    """
    Simple regex-based YAML parser for frontmatter.
    Handles:
      - key: value pairs
      - quoted values (single or double)
      - list items (lines starting with -)
      - multi-line values (indented continuation lines)
    """
    result = {}
    lines = yaml_text.split("\n")
    current_key = None
    current_value = None
    in_list = False
    list_items = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Check if this is a list item (starts with whitespace + -)
        list_match = re.match(r"^\s+-\s*(.*)", line)
        if list_match and current_key is not None:
            if not in_list:
                in_list = True
                list_items = []
            item = list_match.group(1).strip()
            # Strip quotes from list items
            item = _strip_quotes(item)
            list_items.append(item)
            continue

        # Check if this is an indented continuation line (not a new key)
        if re.match(r"^\s+\S", line) and current_key is not None and not in_list:
            # Multi-line continuation
            if current_value is None:
                current_value = ""
            current_value += " " + line.strip()
            continue

        # If we were building a list, save it
        if in_list and current_key is not None:
            result[current_key] = list_items
            in_list = False
            list_items = []
            current_key = None
            current_value = None

        # If we had a previous key with a multi-line value, save it
        if current_key is not None and not in_list:
            result[current_key] = current_value if current_value is not None else ""

        # Try to match a key: value pair
        kv_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)", line)
        if kv_match:
            current_key = kv_match.group(1).strip()
            raw_value = kv_match.group(2).strip()

            if raw_value == "" or raw_value == "|" or raw_value == ">":
                # Value will come on next lines, or it's a list
                current_value = None
                in_list = False
            else:
                current_value = _strip_quotes(raw_value)
                in_list = False
        else:
            # Line doesn't match anything we expect, skip
            continue

    # Save the last key
    if in_list and current_key is not None:
        result[current_key] = list_items
    elif current_key is not None:
        result[current_key] = current_value if current_value is not None else ""

    return result


def _strip_quotes(value: str) -> str:
    """Remove surrounding single or double quotes from a string."""
    if not value:
        return value
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def parse_frontmatter(content: str) -> tuple:
    """
    Extract YAML frontmatter between first pair of --- markers.

    Returns:
        (metadata_dict, body_without_frontmatter)
        If no frontmatter found, returns ({}, content)
    """
    # Match frontmatter: starts at beginning of content with ---
    pattern = r"^---\s*\n(.*?)\n---\s*\n?"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        # Also try with possible leading whitespace/BOM
        pattern_alt = r"^\s*---\s*\n(.*?)\n---\s*\n?"
        match = re.match(pattern_alt, content, re.DOTALL)

    if not match:
        return ({}, content)

    yaml_block = match.group(1)
    body = content[match.end():]

    metadata = _parse_yaml_block(yaml_block)

    return (metadata, body)


def normalize_title(raw_title: str, metadata: dict, domain: str) -> str:
    """
    Normalize the title based on domain-specific rules.

    For HackerRank: if article_slug exists, derive title from it
    (replace hyphens with spaces, strip leading numbers, title case).
    If not, take first sentence/clause of raw_title up to first repeated
    word pattern.

    For others: return raw_title stripped.
    """
    if not raw_title:
        return ""

    if domain == "hackerrank":
        # Prefer article_slug if available
        article_slug = metadata.get("article_slug", "")
        if article_slug:
            # Replace hyphens with spaces
            title = article_slug.replace("-", " ")
            # Strip leading numbers (e.g., "001-some-article" -> "some article")
            title = re.sub(r"^\d+\s*", "", title)
            # Title case
            title = title.strip().title()
            return title

        # Fallback: detect repeated word pattern in concatenated title
        # Split into words and find where repetition starts
        words = raw_title.split()
        if len(words) > 3:
            # Look for a word that appeared earlier, suggesting concatenation
            seen = set()
            cut_index = len(words)
            for i, word in enumerate(words):
                word_lower = word.lower()
                if word_lower in seen and i > 2:
                    cut_index = i
                    break
                seen.add(word_lower)
            title = " ".join(words[:cut_index])
            return title.strip()

        return raw_title.strip()

    # For claude and visa: just strip
    return raw_title.strip()


def strip_images(text: str) -> str:
    """
    Remove markdown image patterns ![alt](url).

    Removes entire lines that are only an image, and inline image references.
    """
    # First, remove standalone image lines (lines that contain only an image)
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.strip()
        # Check if line is ONLY an image (possibly with surrounding whitespace)
        if re.match(r"^!\[.*?\]\(.*?\)\s*$", stripped):
            result_lines.append("")  # Leave empty line
            continue
        # Remove inline images from lines with other content
        cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", line)
        result_lines.append(cleaned)

    return "\n".join(result_lines)


def strip_html_comments(text: str) -> str:
    """
    Remove HTML comments <!-- ... --> including multi-line ones.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def strip_related_articles(text: str) -> str:
    """
    Remove '## Related Articles' section and everything after it.
    Handles both '## Related Articles' and '## Related articles'.
    """
    result = re.sub(r"\n## Related [Aa]rticles.*", "", text, flags=re.DOTALL)
    return result


def strip_artifacts(text: str) -> str:
    """
    Remove common formatting artifacts:
    - \\r\\n -> \\n
    - Trailing backslashes at end of lines
    - Lines that are only whitespace + special characters (no actual word content)
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Remove trailing backslashes (with optional trailing whitespace) per line
    text = re.sub(r"\\\s*$", "", text, flags=re.MULTILINE)

    # Remove lines that contain only whitespace and special characters
    # (no alphanumeric content at all)
    lines = text.split("\n")
    result_lines = []
    for line in lines:
        # Keep empty lines (they'll be normalized later)
        if line.strip() == "":
            result_lines.append(line)
            continue
        # Keep lines that have at least one alphanumeric character
        if re.search(r"[a-zA-Z0-9]", line):
            result_lines.append(line)
        else:
            # Line has only special chars / whitespace — remove it
            result_lines.append("")

    return "\n".join(result_lines)


def normalize_whitespace(text: str) -> str:
    """
    - Collapse 3+ consecutive newlines into exactly 2 (one blank line).
    - Strip trailing whitespace per line.
    - Strip leading/trailing whitespace from entire text.
    """
    # Strip trailing whitespace per line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines into \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from entire document
    text = text.strip()

    return text


def clean_document(content: str, domain: str) -> tuple:
    """
    Full cleaning pipeline for a single document.

    Chains: parse_frontmatter -> normalize_title -> strip_images ->
            strip_html_comments -> strip_related_articles ->
            strip_artifacts -> normalize_whitespace

    Args:
        content: Raw markdown content of the document.
        domain: One of 'hackerrank', 'claude', 'visa'.

    Returns:
        (metadata_with_clean_title, cleaned_body)
    """
    # Step 1: Parse frontmatter
    metadata, body = parse_frontmatter(content)

    # Step 2: Normalize title and add to metadata
    raw_title = metadata.get("title", "")
    clean_title = normalize_title(raw_title, metadata, domain)
    metadata["clean_title"] = clean_title

    # Step 3: Strip images
    body = strip_images(body)

    # Step 4: Strip HTML comments
    body = strip_html_comments(body)

    # Step 5: Strip related articles section
    body = strip_related_articles(body)

    # Step 6: Strip artifacts
    body = strip_artifacts(body)

    # Step 7: Normalize whitespace
    body = normalize_whitespace(body)

    return (metadata, body)


def detect_stub(content: str) -> bool:
    """
    Detect if a document is a stub (fewer than 5 non-blank lines of body).

    Args:
        content: Raw markdown content.

    Returns:
        True if the document body has fewer than 5 non-blank lines.
    """
    _, body = parse_frontmatter(content)

    non_blank_lines = [line for line in body.split("\n") if line.strip()]

    return len(non_blank_lines) < 5


# --- Cloudflare email redaction cleanup ---

def strip_cloudflare_email_protection(text: str) -> str:
    """
    Remove Cloudflare email protection artifacts.
    Replaces [email\xa0protected] and /cdn-cgi/l/email-protection links
    with a placeholder.
    """
    # Remove links with email protection
    text = re.sub(
        r'\[([^\]]*?)\]\(/cdn-cgi/l/email-protection[^)]*\)',
        r'\1',
        text
    )
    # Replace the [email protected] placeholder text
    text = re.sub(
        r'\[email\s*protected\]',
        '[email]',
        text
    )
    # Also handle the HTML anchor variant
    text = re.sub(
        r'<a[^>]*href="[^"]*cdn-cgi/l/email-protection[^"]*"[^>]*>.*?</a>',
        '[email]',
        text
    )

    return text


def clean_document_full(content: str, domain: str) -> tuple:
    """
    Extended cleaning pipeline that also handles Cloudflare email protection.
    Use this instead of clean_document for the Visa corpus.

    Args:
        content: Raw markdown content.
        domain: One of 'hackerrank', 'claude', 'visa'.

    Returns:
        (metadata_with_clean_title, cleaned_body)
    """
    metadata, body = clean_document(content, domain)

    # Additional step for Visa corpus: strip Cloudflare email artifacts
    if domain == "visa":
        body = strip_cloudflare_email_protection(body)

    return (metadata, body)


if __name__ == "__main__":
    # Quick self-test
    sample = """---
title: Test Article Title
source_url: https://example.com/test
article_slug: my-test-article
last_updated_exact: 2025-01-01
breadcrumbs:
  - Home
  - Support
  - Test
---

# My Test Article

This is the body of the article.

![some image](https://cdn.example.com/very-long-url-here.png)

<!-- spacer comment -->

Some more content here.\\

## Related Articles

- [Link 1](url1)
- [Link 2](url2)
"""
    metadata, body = clean_document(sample, "hackerrank")
    print(f"Clean title: {metadata.get('clean_title')}")
    print(f"Body preview: {body[:200]}")
    print(f"Is stub: {detect_stub(sample)}")
