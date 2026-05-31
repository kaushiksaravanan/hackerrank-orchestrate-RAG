"""End-to-end integration tests on REAL sample data with REAL API calls.
These tests REQUIRE contradictory outputs — no single stub can pass all of them.
Mark as slow since they involve API calls."""
import pytest
import re

# Mark all tests in this module as requiring API access
pytestmark = pytest.mark.integration


def test_sample_site_down(retriever_instance, llm_client):
    """Sample #2: 'site is down' MUST be escalated/bug. Contradicts test_sample_iron_man."""
    from agent import process_ticket
    result = process_ticket('site is down & none of the pages are accessible', '', 'None', retriever_instance, llm_client)
    assert result['status'].lower() == 'escalated', f"Expected escalated, got {result['status']}"
    assert result['request_type'] == 'bug', f"Expected bug, got {result['request_type']}"


def test_sample_iron_man(retriever_instance, llm_client):
    """Sample #7: Iron Man MUST be replied/invalid. Contradicts test_sample_site_down."""
    from agent import process_ticket
    result = process_ticket('What is the name of the actor in Iron Man?', 'Urgent, please help', 'None', retriever_instance, llm_client)
    assert result['status'].lower() == 'replied', f"Expected replied, got {result['status']}"
    assert result['request_type'] == 'invalid', f"Expected invalid, got {result['request_type']}"


def test_sample_thank_you(retriever_instance, llm_client):
    """Sample #10: Thank you MUST be replied/invalid."""
    from agent import process_ticket
    result = process_ticket('Thank you for helping me', '', 'None', retriever_instance, llm_client)
    assert result['status'].lower() == 'replied'
    assert result['request_type'] == 'invalid'


def test_sample_visa_lost_card(retriever_instance, llm_client):
    """Sample #9: Lost Visa card must be replied with phone number 000-800-100-1219."""
    from agent import process_ticket
    result = process_ticket('Where can I report a lost or stolen Visa card from India?', 'Card stolen', 'Visa', retriever_instance, llm_client)
    assert result['status'].lower() == 'replied'
    # Response must contain the actual phone number from corpus
    assert '1219' in result['response'] or '000-800' in result['response'], \
        f"Expected phone 000-800-100-1219 in response: {result['response'][:200]}"


def test_sample_visa_cheques(retriever_instance, llm_client):
    """Sample #8: Stolen cheques must be replied with Citicorp phone number."""
    from agent import process_ticket
    result = process_ticket("I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?", '', 'Visa', retriever_instance, llm_client)
    assert result['status'].lower() == 'replied'
    # Must contain a phone number
    assert re.search(r'\d{3}.*\d{4}', result['response']), \
        f"No phone number in response: {result['response'][:200]}"


def test_output_schema(retriever_instance, llm_client):
    """Output must have all required fields with allowed values."""
    from agent import process_ticket
    # Use a simple ticket that should work via fast path
    result = process_ticket('Thank you', '', 'None', retriever_instance, llm_client)
    required_keys = {'status', 'product_area', 'response', 'justification', 'request_type'}
    assert required_keys.issubset(set(result.keys())), f"Missing keys: {required_keys - set(result.keys())}"
    assert result['status'].lower() in ('replied', 'escalated')
    assert result['request_type'] in ('product_issue', 'feature_request', 'bug', 'invalid')


def test_no_empty_responses(retriever_instance, llm_client):
    """Process 3 diverse tickets — none should have empty response."""
    from agent import process_ticket
    tickets = [
        ('Thank you for helping me', '', 'None'),
        ('site is down', '', 'None'),
        ('What is the name of the actor in Iron Man?', 'Urgent', 'None'),
    ]
    for issue, subject, company in tickets:
        result = process_ticket(issue, subject, company, retriever_instance, llm_client)
        assert result.get('response') not in (None, ''), f"Empty response for: {issue[:50]}"
