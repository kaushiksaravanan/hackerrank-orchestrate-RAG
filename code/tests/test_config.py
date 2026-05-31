"""Tests that config points to real files and has correct taxonomy."""
import os
from pathlib import Path


def test_data_dir_exists():
    from config import DATA_DIR
    assert DATA_DIR.exists()
    subdirs = set(os.listdir(DATA_DIR))
    assert {'hackerrank', 'claude', 'visa'}.issubset(subdirs)


def test_support_dir_has_csv():
    from config import SUPPORT_DIR
    assert (SUPPORT_DIR / 'support_tickets.csv').exists()


def test_product_areas_taxonomy():
    from prompts import PRODUCT_AREA_TAXONOMY
    assert 'hackerrank' in PRODUCT_AREA_TAXONOMY
    assert 'screen' in PRODUCT_AREA_TAXONOMY['hackerrank']
    assert 'visa' in PRODUCT_AREA_TAXONOMY
    assert 'travel_support' in PRODUCT_AREA_TAXONOMY['visa']
    assert 'claude' in PRODUCT_AREA_TAXONOMY
    assert 'privacy' in PRODUCT_AREA_TAXONOMY['claude']
    # Each domain has at least 5 categories
    for domain, areas in PRODUCT_AREA_TAXONOMY.items():
        assert len(areas) >= 5, f"{domain} has too few categories"


def test_api_key_set():
    from config import ANTHROPIC_API_KEY
    assert isinstance(ANTHROPIC_API_KEY, str)
    assert len(ANTHROPIC_API_KEY) > 0


def test_domains_match_disk():
    from config import DOMAINS, DATA_DIR
    for d in DOMAINS:
        assert (DATA_DIR / d).is_dir(), f"Domain dir {d} missing from disk"


def test_skip_files():
    from config import SKIP_FILES
    assert 'consumer.md' in SKIP_FILES
    assert 'merchant.md' in SKIP_FILES


def test_params_positive():
    from config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, BM25_TOP_K, VECTOR_TOP_K, FINAL_TOP_K
    assert MAX_CHUNK_TOKENS == 512
    assert MIN_CHUNK_TOKENS > 0
    assert BM25_TOP_K > 0
    assert VECTOR_TOP_K > 0
    assert FINAL_TOP_K > 0
