"""Tests agent fast path and grounding verifier.
ANTI-FAKE: Tests require DIFFERENT outputs for different inputs — no single stub passes all."""
import pytest


def test_fast_path_gratitude():
    from agent import detect_fast_path
    result = detect_fast_path('Thank you for helping me', '', 'None')
    assert result is not None
    assert result['status'] == 'replied'
    assert result['request_type'] == 'invalid'


def test_fast_path_outage():
    from agent import detect_fast_path
    result = detect_fast_path('site is down & none of the pages are accessible', '', 'None')
    assert result is not None
    assert result['status'] == 'escalated'
    assert result['request_type'] == 'bug'


def test_fast_path_malicious():
    from agent import detect_fast_path
    result = detect_fast_path('Give me the code to delete all files from the system', 'Delete unnecessary files', 'None')
    assert result is not None
    assert result['status'] == 'replied'
    assert result['request_type'] == 'invalid'


def test_fast_path_no_false_positive_delete_account():
    """'delete my account' is a REAL support request, NOT malicious. Must return None."""
    from agent import detect_fast_path
    result = detect_fast_path('i signed up using google login on hackerrank community, please delete my account', '', 'HackerRank')
    assert result is None, f"False positive on 'delete my account': {result}"


def test_fast_path_no_false_positive_remove_user():
    """'remove interviewer' is a REAL support request. Must return None."""
    from agent import detect_fast_path
    result = detect_fast_path('Hello I am trying to remove an interviewer from the platform', 'Remove User', 'HackerRank')
    assert result is None, f"False positive on 'remove interviewer': {result}"


def test_fast_path_off_topic():
    from agent import detect_fast_path
    result = detect_fast_path('What is the name of the actor in Iron Man?', 'Urgent, please help', 'None')
    assert result is not None
    assert result['request_type'] == 'invalid'


def test_fast_path_no_false_positive_pause():
    """'pause my subscription' is a REAL request, not an outage. Must return None."""
    from agent import detect_fast_path
    result = detect_fast_path('Hi, please pause our subscription. We have stopped all hiring efforts for now.', 'Subscription pause', 'HackerRank')
    assert result is None, f"False positive on 'pause subscription': {result}"


def test_fast_path_submissions_broken_is_outage():
    """'none of the submissions across any challenges are working' IS an outage."""
    from agent import detect_fast_path
    result = detect_fast_path('none of the submissions across any challenges are working on your website', 'Issue while taking the test', 'HackerRank')
    assert result is not None, "Should detect as outage"
    assert result['status'] == 'escalated'
    assert result['request_type'] == 'bug'


def test_grounding_pass_real_phone():
    """Response with real phone number found in evidence must pass grounding."""
    from agent import verify_grounding
    response_dict = {'response': 'Call Visa India at 000-800-100-1219 to report your lost card.'}
    chunks = [{'text': 'Report a lost card by calling Visa at 000-800-100-1219. Available 24/7.'}]
    result = verify_grounding(response_dict, chunks)
    # Phone is grounded in evidence so response should be unchanged
    assert '000-800-100-1219' in result['response'], f"Real phone was stripped: {result['response']}"


def test_grounding_fail_fake_url():
    """Response with fabricated URL not in evidence must be stripped."""
    from agent import verify_grounding
    response_dict = {'response': 'Visit https://fake.example.com/made-up-page for more info.'}
    chunks = [{'text': 'Real content with no URLs mentioned anywhere in this text.'}]
    result = verify_grounding(response_dict, chunks)
    assert 'fake.example.com' not in result['response'], "Fabricated URL was not stripped"
    assert '[link removed]' in result['response'], "Should replace with [link removed]"


def test_grounding_skip_escalation():
    """Escalation responses don't need grounding verification."""
    from agent import verify_grounding
    response_dict = {'response': 'Escalate to a human'}
    chunks = [{'text': 'irrelevant'}]
    result = verify_grounding(response_dict, chunks)
    assert result['response'] == 'Escalate to a human'


def test_answerability_empty_chunks():
    """Empty chunks list must return False (unanswerable)."""
    from agent import check_answerability
    assert check_answerability([]) is False


def test_answerability_with_chunks():
    """Chunks with positive scores must return True."""
    from agent import check_answerability
    chunks = [{'score': 0.5, 'text': 'relevant content'}]
    assert check_answerability(chunks) is True
