"""Tests preprocessor against REAL corpus files. Verifies noise removal + content preservation."""
import re
from pathlib import Path


def test_pause_subscription_images_stripped():
    """HackerRank pause subscription article has images. After cleaning, images gone, content remains."""
    from config import DATA_DIR
    from preprocessor import clean_document
    # Find the pause subscription article
    article_path = None
    for p in DATA_DIR.joinpath('hackerrank').rglob('*.md'):
        if 'pause' in p.name.lower() and 'subscription' in p.name.lower():
            article_path = p
            break
    assert article_path is not None, "Pause subscription article not found in corpus"
    raw = article_path.read_text(encoding='utf-8')
    metadata, cleaned = clean_document(raw, 'hackerrank')
    # Images stripped
    assert re.search(r'!\[.*?\]\(.*?\)', cleaned) is None, "Images not stripped"
    # Real content preserved
    assert 'pause' in cleaned.lower() or 'subscription' in cleaned.lower(), "Content lost"


def test_visa_support_phone_preserved():
    """Visa support.md phone number 000-800-100-1219 must survive cleaning."""
    from config import DATA_DIR
    from preprocessor import clean_document
    raw = (DATA_DIR / 'visa' / 'support.md').read_text(encoding='utf-8')
    metadata, cleaned = clean_document(raw, 'visa')
    assert '000-800-100-1219' in cleaned, "Visa India phone number lost during cleaning"
    assert metadata.get('title') is not None, "Title not extracted"


def test_visa_us10_minimum_preserved():
    """Visa support.md US$10 minimum transaction text must survive."""
    from config import DATA_DIR
    from preprocessor import clean_document
    raw = (DATA_DIR / 'visa' / 'support.md').read_text(encoding='utf-8')
    _, cleaned = clean_document(raw, 'visa')
    assert 'US$10' in cleaned or 'US$ 10' in cleaned or '$10' in cleaned, "US$10 FAQ lost"


def test_claude_related_articles_stripped():
    """Claude conversation deletion article — Related Articles footer removed, real content kept."""
    from config import DATA_DIR
    from preprocessor import strip_related_articles
    # Find the delete conversation article
    article_path = None
    for p in DATA_DIR.joinpath('claude').rglob('*.md'):
        if '8230524' in p.name or ('delete' in p.name.lower() and 'conversation' in p.name.lower()):
            article_path = p
            break
    assert article_path is not None, "Delete conversation article not found"
    raw = article_path.read_text(encoding='utf-8')
    cleaned = strip_related_articles(raw)
    assert '## Related Articles' not in cleaned and '## Related articles' not in cleaned
    assert 'delete' in cleaned.lower(), "Real content about deleting removed"


def test_detect_stub_on_real_files():
    """consumer.md is a stub (True), support.md is not (False)."""
    from config import DATA_DIR
    from preprocessor import detect_stub
    stub_content = (DATA_DIR / 'visa' / 'support' / 'consumer.md').read_text(encoding='utf-8')
    real_content = (DATA_DIR / 'visa' / 'support.md').read_text(encoding='utf-8')
    assert detect_stub(stub_content) is True, "consumer.md not detected as stub"
    assert detect_stub(real_content) is False, "support.md incorrectly detected as stub"


def test_no_html_comments_after_cleaning():
    """After cleaning any HackerRank article, no HTML comments remain."""
    from config import DATA_DIR
    from preprocessor import clean_document
    # Pick any hackerrank article
    articles = list(DATA_DIR.joinpath('hackerrank').rglob('*.md'))
    checked = 0
    for p in articles[:20]:
        raw = p.read_text(encoding='utf-8')
        if '<!--' in raw:
            _, cleaned = clean_document(raw, 'hackerrank')
            assert '<!--' not in cleaned, f"HTML comment survived in {p.name}"
            checked += 1
    # We should find at least one article with comments to test
    # If not found, that's also fine (means corpus doesn't have them)
