"""Tests that prompts contain required calibration data and rules."""


def test_few_shot_count():
    from prompts import FEW_SHOT_EXAMPLES
    assert len(FEW_SHOT_EXAMPLES) == 10, f"Expected 10 few-shot examples, got {len(FEW_SHOT_EXAMPLES)}"


def test_few_shot_site_down():
    """Sample #2: 'site is down' must be Escalated/bug with empty product_area."""
    from prompts import FEW_SHOT_EXAMPLES
    sample = next(s for s in FEW_SHOT_EXAMPLES if 'site is down' in s.get('issue', '').lower())
    assert sample['status'] == 'escalated'
    assert sample['request_type'] == 'bug'
    assert sample['product_area'] == ''


def test_few_shot_iron_man():
    """Sample #7: Iron Man must be invalid."""
    from prompts import FEW_SHOT_EXAMPLES
    sample = next(s for s in FEW_SHOT_EXAMPLES if 'iron man' in s.get('issue', '').lower())
    assert sample['request_type'] == 'invalid'
    assert sample['product_area'] == 'conversation_management'


def test_few_shot_thank_you():
    """Sample #10: Thank you must be Replied/invalid with empty product_area."""
    from prompts import FEW_SHOT_EXAMPLES
    sample = next(s for s in FEW_SHOT_EXAMPLES if 'thank you' in s.get('issue', '').lower())
    assert sample['status'] == 'replied'
    assert sample['request_type'] == 'invalid'
    assert sample['product_area'] == ''


def test_few_shot_visa_card():
    """Sample #9: Lost Visa card must have product_area=general_support."""
    from prompts import FEW_SHOT_EXAMPLES
    sample = next(s for s in FEW_SHOT_EXAMPLES if 'lost or stolen visa card' in s.get('issue', '').lower()
                  or 'report a lost' in s.get('issue', '').lower())
    assert sample['product_area'] == 'general_support'


def test_system_prompt_contains_allowed_values():
    from prompts import SYSTEM_PROMPT
    for value in ['replied', 'escalated', 'product_issue', 'feature_request', 'bug', 'invalid']:
        assert value in SYSTEM_PROMPT, f"'{value}' missing from SYSTEM_PROMPT"


def test_system_prompt_adversarial_defense():
    from prompts import SYSTEM_PROMPT
    defense_keywords = ['injection', 'internal rules', 'system prompt', 'ignore']
    found = sum(1 for k in defense_keywords if k.lower() in SYSTEM_PROMPT.lower())
    assert found >= 2, f"Only {found}/4 adversarial defense keywords found in SYSTEM_PROMPT"


def test_format_prompt_includes_evidence():
    from prompts import format_prompt
    result = format_prompt('test issue text', 'test subject', 'HackerRank',
                          [{'text': 'evidence chunk content', 'title': 'Article Title', 'section': 'Section Name', 'domain': 'hackerrank'}])
    assert 'test issue text' in result
    assert 'evidence chunk content' in result
    assert 'Article Title' in result


def test_taxonomy_completeness():
    from prompts import PRODUCT_AREA_TAXONOMY
    assert 'screen' in PRODUCT_AREA_TAXONOMY['hackerrank']
    assert 'interviews' in PRODUCT_AREA_TAXONOMY['hackerrank']
    assert 'community' in PRODUCT_AREA_TAXONOMY['hackerrank']
    assert 'general_support' in PRODUCT_AREA_TAXONOMY['visa']
    assert 'travel_support' in PRODUCT_AREA_TAXONOMY['visa']
    assert 'privacy' in PRODUCT_AREA_TAXONOMY['claude']


def test_few_shot_has_justification():
    """Every few-shot example must have a non-empty justification field."""
    from prompts import FEW_SHOT_EXAMPLES
    for i, example in enumerate(FEW_SHOT_EXAMPLES):
        assert 'justification' in example, f"Few-shot example {i} missing 'justification' key"
        assert example['justification'].strip(), f"Few-shot example {i} has empty justification"
