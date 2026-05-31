"""
Core agent module for the support triage agent.

Contains the fast-path classifier, LLM generation, grounding verifier,
and main orchestration function (process_ticket).
"""

import json
import logging
import re

from llm import LLMClient
from prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, format_prompt
from retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Few-Shot Message Builder ────────────────────────────────────────────────


def _build_few_shot_messages() -> list[dict]:
    """
    Build alternating user/assistant message pairs from FEW_SHOT_EXAMPLES.

    Each user message contains the ticket info (without evidence chunks),
    each assistant message contains the expected JSON output.
    """
    messages: list[dict] = []
    for ex in FEW_SHOT_EXAMPLES:
        company = ex.get("company", "None")
        subject = ex.get("subject", "")
        issue = ex.get("issue", "")

        user_msg = (
            f"## Support Ticket\n"
            f"**Company:** {company}\n"
            f"**Subject:** {subject}\n" if subject else f"**Subject:** (none)\n"
            f"**Issue:**\n{issue}\n\n"
            f"## Retrieved Evidence\n"
            f"(Evidence retrieved from support corpus)\n\n"
            f"## Instructions\n"
            f"Respond with ONLY a valid JSON object matching the schema "
            f"described in the system prompt."
        )
        # Fix: build user_msg properly (the ternary above breaks the f-string chain)
        user_msg = (
            f"## Support Ticket\n"
            f"**Company:** {company}\n"
            + (f"**Subject:** {subject}\n" if subject else "**Subject:** (none)\n")
            + f"**Issue:**\n{issue}\n\n"
            f"## Retrieved Evidence\n"
            f"(Evidence retrieved from support corpus)\n\n"
            f"## Instructions\n"
            f"Respond with ONLY a valid JSON object matching the schema "
            f"described in the system prompt."
        )
        messages.append({"role": "user", "content": user_msg})

        response_json = {
            "status": ex["status"],
            "product_area": ex.get("product_area", ""),
            "response": ex.get("response", ""),
            "justification": ex.get("justification", ""),
            "request_type": ex.get("request_type", ""),
        }
        messages.append({"role": "assistant", "content": json.dumps(response_json)})

    return messages


# Cache at module level (built once)
_FEW_SHOT_MESSAGES: list[dict] | None = None


def _get_few_shot_messages() -> list[dict]:
    global _FEW_SHOT_MESSAGES
    if _FEW_SHOT_MESSAGES is None:
        _FEW_SHOT_MESSAGES = _build_few_shot_messages()
    return _FEW_SHOT_MESSAGES


# ─── Fast-Path Classifier ────────────────────────────────────────────────────


def detect_fast_path(issue: str, subject: str, company: str | None) -> dict | None:
    """
    Rule-based fast path for obvious cases that don't need retrieval.
    Returns a result dict or None if retrieval is needed.

    First match wins — order matters.
    """
    issue_stripped = issue.strip()

    # A) GRATITUDE — short messages only
    if len(issue_stripped) < 100:
        gratitude_pattern = re.compile(
            r"^\s*(thank|thanks|thx|cheers|happy to help|appreciate)",
            re.IGNORECASE,
        )
        if gratitude_pattern.search(issue_stripped):
            return {
                "status": "replied",
                "request_type": "invalid",
                "product_area": "",
                "response": "Happy to help",
                "justification": "User expressed gratitude with no actionable request.",
            }

    # B) PLATFORM OUTAGE — only for platform-wide reports
    # First exclude single-user / single-project language
    single_user_pattern = re.compile(
        r"\b(my project|my application|my app|my account|my test|my submission|"
        r"i am facing|i'm facing|i have|i had|in my|"
        r"single|one challenge|specific|particular|"
        r"pause|subscription|cancel|delete my|remove my|"
        r"bedrock|aws|api key|endpoint|integration)\b",
        re.IGNORECASE,
    )
    if not single_user_pattern.search(issue_stripped):
        outage_pattern_1 = re.compile(
            r"(site|website|service|platform|everything|all\s+(requests|submissions|pages))"
            r".*(down|not\s+working|failing|broken|inaccessible)",
            re.IGNORECASE | re.DOTALL,
        )
        outage_pattern_2 = re.compile(
            r"(down|not\s+working|failing|broken)"
            r".*(site|website|service|platform|everything|all)",
            re.IGNORECASE | re.DOTALL,
        )
        outage_pattern_3 = re.compile(
            r"none of the\s+(submissions|pages|requests|tests|challenges)"
            r".*(working|loading|accessible)",
            re.IGNORECASE | re.DOTALL,
        )
        if (
            outage_pattern_1.search(issue_stripped)
            or outage_pattern_2.search(issue_stripped)
            or outage_pattern_3.search(issue_stripped)
        ):
            return {
                "status": "escalated",
                "request_type": "bug",
                "product_area": "",
                "response": "Escalate to a human",
                "justification": "Platform-wide outage reported requiring immediate human investigation.",
            }

    # C) MALICIOUS / CODE REQUEST
    legitimate_deletion_pattern = re.compile(
        r"(delete my account|delete my data|remove my account|remove user|"
        r"remove interviewer|delete.*conversation|delete.*profile|"
        r"remove.*employee|remove.*member)",
        re.IGNORECASE,
    )
    if not legitimate_deletion_pattern.search(issue_stripped):
        malicious_pattern_1 = re.compile(
            r"(give me|write|generate|provide)"
            r".*(code|script|program|command)"
            r".*(delete|hack|exploit|remove all|destroy)",
            re.IGNORECASE | re.DOTALL,
        )
        malicious_pattern_2 = re.compile(
            r"(delete all files|rm -rf|format.*hard drive|drop.*database)",
            re.IGNORECASE,
        )
        if malicious_pattern_1.search(issue_stripped) or malicious_pattern_2.search(
            issue_stripped
        ):
            return {
                "status": "replied",
                "request_type": "invalid",
                "product_area": "",
                "response": "I am sorry, this is out of scope from my capabilities",
                "justification": "Request is for potentially harmful code generation, not a support issue.",
            }

    # D) OFF-TOPIC — broader coverage
    offtopic_pattern = re.compile(
        r"\b(iron man|avengers|movie|actor|actress|recipe|weather forecast|"
        r"sports score|stock price|cryptocurrency|bitcoin|pokemon|"
        r"game of thrones|netflix|spotify|instagram|tiktok|"
        r"who is the president|capital of|population of)\b",
        re.IGNORECASE,
    )
    if offtopic_pattern.search(issue_stripped):
        # Ensure it's not also mentioning a supported product
        product_mention = re.compile(
            r"\b(hackerrank|claude|anthropic|visa)\b", re.IGNORECASE,
        )
        if not product_mention.search(issue_stripped):
            return {
                "status": "replied",
                "request_type": "invalid",
                "product_area": "",
                "response": "I am sorry, this is out of scope from my capabilities",
                "justification": "Request is unrelated to any supported product.",
            }

    # No fast-path match — needs retrieval
    return None


# ─── Answerability Check ─────────────────────────────────────────────────────


def check_answerability(chunks: list[dict], threshold: float = 0.015) -> bool:
    """
    Check if retrieved chunks provide enough evidence to answer.
    Returns True if answerable, False if should escalate.
    """
    if not chunks:
        return False

    best_score = max(chunk.get("score", 0.0) for chunk in chunks)
    if best_score < threshold:
        return False

    return True


# ─── LLM Generation ──────────────────────────────────────────────────────────


def generate_response(
    issue: str,
    subject: str,
    company: str,
    chunks: list[dict],
    llm_client: LLMClient,
) -> dict:
    """
    Call LLM via the unified LLMClient to generate a structured JSON response.
    Includes few-shot examples for format alignment.
    Retries up to 2 times if JSON parsing fails.
    """
    user_message = format_prompt(issue, subject, company, chunks)

    # Build messages: few-shot examples + actual query
    few_shot = _get_few_shot_messages()
    messages = few_shot + [{"role": "user", "content": user_message}]

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            raw_text = llm_client.generate(
                system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=2000,
                temperature=0,
            )

            # Parse JSON — handle markdown code blocks
            json_text = raw_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[len("```json"):].strip()
            elif json_text.startswith("```"):
                json_text = json_text[len("```"):].strip()
            if json_text.endswith("```"):
                json_text = json_text[:-len("```")].strip()

            result = json.loads(json_text)

            # Validate required fields
            valid_statuses = ("replied", "escalated")
            valid_request_types = ("product_issue", "feature_request", "bug", "invalid")

            if result.get("status") not in valid_statuses:
                raise ValueError(f"Invalid status: {result.get('status')}")
            if result.get("request_type") not in valid_request_types:
                raise ValueError(f"Invalid request_type: {result.get('request_type')}")

            return {
                "status": result["status"],
                "product_area": result.get("product_area", ""),
                "response": result.get("response", ""),
                "justification": result.get("justification", ""),
                "request_type": result["request_type"],
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d failed to parse LLM response: %s",
                attempt + 1, max_retries, e,
            )
            continue
        except Exception as e:
            last_error = e
            logger.error("LLM call failed on attempt %d: %s", attempt + 1, e)
            break

    # All retries exhausted
    logger.error("All retries failed for generate_response. Last error: %s", last_error)
    return {
        "status": "escalated",
        "request_type": "bug",
        "product_area": "",
        "response": "Unable to process this request. Escalating to human support.",
        "justification": f"LLM generation failed after {max_retries} attempts: {str(last_error)[:100]}",
    }


# ─── Grounding Verifier ──────────────────────────────────────────────────────


def verify_grounding(response_dict: dict, chunks: list[dict]) -> dict:
    """
    Verify that factual claims (URLs, phone numbers, emails) in the response
    are grounded in the evidence chunks.

    Returns the (possibly modified) response dict:
    - Strips ungrounded URLs/phones/emails from the response text
    - Adds a warning to justification if items were stripped
    """
    response_text = response_dict.get("response", "")

    # Short escalation / fast-path responses — skip verification
    skip_phrases = (
        "escalate to a human",
        "happy to help",
        "out of scope from my capabilities",
        "unable to process",
    )
    if response_text.lower().strip() in skip_phrases or len(response_text) < 30:
        return response_dict

    # Extract factual claims from the response
    urls = re.findall(r"https?://[^\s\)\"']+", response_text)
    phones = re.findall(r"[\+]?[\d][\d\s\-\(\)]{6,}[\d]", response_text)
    emails = re.findall(r"[\w.\-]+@[\w.\-]+\.\w+", response_text)

    all_items = urls + phones + emails
    if not all_items:
        return response_dict

    # Build combined evidence string
    evidence = "\n".join(chunk.get("text", "") for chunk in chunks)
    evidence_digits = re.sub(r"[^\d]", "", evidence)

    ungrounded: list[str] = []
    modified_response = response_text

    for item in urls:
        cleaned = item.rstrip(".,;:!?)")
        if cleaned not in evidence:
            ungrounded.append(cleaned)
            modified_response = modified_response.replace(item, "[link removed]")

    for item in phones:
        digits = re.sub(r"[^\d]", "", item)
        if len(digits) >= 7 and digits not in evidence_digits:
            ungrounded.append(item)
            modified_response = modified_response.replace(item, "[number removed]")

    for item in emails:
        if item not in evidence:
            ungrounded.append(item)
            modified_response = modified_response.replace(item, "[email removed]")

    if ungrounded:
        logger.warning("Stripped %d ungrounded items: %s", len(ungrounded), ungrounded)
        response_dict = dict(response_dict)  # copy to avoid mutating original
        response_dict["response"] = modified_response
        if len(ungrounded) > len(all_items) // 2:
            response_dict["justification"] = (
                response_dict.get("justification", "")
                + f" [Note: {len(ungrounded)} unverified factual claims removed from response.]"
            )

    return response_dict


# ─── Main Orchestration ──────────────────────────────────────────────────────


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    retriever: Retriever,
    llm_client: LLMClient,
    **kwargs,
) -> dict:
    """
    Full pipeline for one support ticket.

    Steps:
      1. Normalize inputs
      2. Fast-path detection
      3. Retrieve evidence chunks
      4. Check answerability
      5. Generate LLM response
      6. Verify grounding
      7. Return result
    """
    try:
        # 1. Normalize inputs
        issue = (issue or "").strip()
        subject = (subject or "").strip()
        if company is None or str(company).strip().lower() in ("", "none", "nan"):
            company = "None"
        else:
            company = str(company).strip()

        # 2. Fast-path detection
        fast_result = detect_fast_path(
            issue, subject, company if company != "None" else None,
        )
        if fast_result is not None:
            return fast_result

        # 3. Retrieve evidence chunks
        chunks = retriever.retrieve(issue, company, top_k=10)

        # 4. Check answerability
        if not check_answerability(chunks):
            return {
                "status": "escalated",
                "request_type": "product_issue",
                "product_area": "",
                "response": "Escalate to a human",
                "justification": "Insufficient evidence in the knowledge base to provide a reliable answer.",
            }

        # 5. Generate LLM response
        result = generate_response(issue, subject, company, chunks, llm_client)

        # 6. Verify grounding (strips ungrounded items, may modify response)
        result = verify_grounding(result, chunks)

        # 7. Return result
        return result

    except Exception as e:
        logger.error("process_ticket failed: %s", e, exc_info=True)
        return {
            "status": "escalated",
            "request_type": "bug",
            "product_area": "",
            "response": "Unable to process this request. Escalating to human support.",
            "justification": f"Processing error: {str(e)[:100]}",
        }
