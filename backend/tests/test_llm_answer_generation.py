"""
Tests for Milestone 4: LLMAnswerGenerationProvider, grounding prompt,
citation-id resolution/safety, and fallback behavior.

NO test in this file makes a real network call. All "LLM calls" are a
FakeOpenAIClient that mimics the shape of `openai`'s
`client.chat.completions.create(...)` response — see FakeOpenAIClient
below. Real-call verification only happens in scripts/llm_rag_demo.py,
run manually and only when OPENAI_API_KEY is actually configured.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from app.services.answer_generation import (
    CORPUS_SCOPE_NOTE,
    ContextItem,
    ExtractiveAnswerGenerationProvider,
    GeneratedAnswer,
    LLMAnswerGenerationProvider,
    build_context_prompt,
    build_grounding_system_prompt,
    get_default_answer_provider,
)

SAMPLE_CONTEXT = [
    ContextItem(
        index=1,
        standard_number="IS 9999",
        standard_title="Sample Standard",
        document_title="Sample QCO Circular",
        document_type="QCO",
        clause_number="3",
        clause_type="CLAUSE",
        clause_text="All manufacturers must obtain BIS certification before sale.",
        page_number=2,
        source_url="https://example.invalid/doc.pdf",
        relevance_score=0.61,
        retrieval_method="vector",
    ),
    ContextItem(
        index=2,
        standard_number="IS 9999",
        standard_title="Sample Standard",
        document_title="Sample Standard Document",
        document_type="STANDARD",
        clause_number="4.1",
        clause_type="CLAUSE",
        clause_text="Earthing requirements for household electrical appliances.",
        page_number=3,
        source_url="https://example.invalid/standard.pdf",
        relevance_score=0.55,
        retrieval_method="vector",
    ),
]


class _FakeMessage:
    def __init__(self, content: Optional[str]):
        self.content = content


class _FakeChoice:
    def __init__(self, content: Optional[str]):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: Optional[str]):
        self.choices = [_FakeChoice(content)]


class FakeOpenAIClient:
    """
    Minimal stand-in for `openai.OpenAI()`. Records every call so tests
    can assert exactly what was sent (system prompt, question, context),
    and returns a scripted response (or raises a scripted exception).
    """

    def __init__(self, response_content: Optional[str] = None, raise_exc: Optional[Exception] = None):
        self._response_content = response_content
        self._raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []

        class _Completions:
            def create(inner_self, **kwargs):
                self.calls.append(kwargs)
                if self._raise_exc is not None:
                    raise self._raise_exc
                return _FakeResponse(self._response_content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _make_provider(client, **kwargs) -> LLMAnswerGenerationProvider:
    return LLMAnswerGenerationProvider(api_key="fake-key-for-tests", client=client, **kwargs)


# ---------------------------------------------------------------------------
# 1. Provider interface compatibility
# ---------------------------------------------------------------------------

def test_llm_provider_implements_same_interface_as_extractive():
    llm_provider = _make_provider(FakeOpenAIClient(response_content=json.dumps(
        {"answer": "ok", "citation_ids": ["SOURCE_1"]}
    )))
    extractive = ExtractiveAnswerGenerationProvider()

    for provider in (llm_provider, extractive):
        result = provider.generate("question", SAMPLE_CONTEXT)
        assert isinstance(result, GeneratedAnswer)
        assert isinstance(result.answer_text, str)
        assert isinstance(result.grounded, bool)
        assert isinstance(result.cited_indices, list)


# ---------------------------------------------------------------------------
# 2 & 3. Correct question / context passed to provider; 4. system prompt present
# ---------------------------------------------------------------------------

def test_correct_question_and_context_sent_to_llm_with_grounding_system_prompt():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"answer": "Certification is required. [SOURCE_1]", "citation_ids": ["SOURCE_1"]})
    )
    provider = _make_provider(fake_client, model="gpt-4o-mini", temperature=0.0)

    provider.generate("Do appliances need certification?", SAMPLE_CONTEXT)

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}

    messages = call["messages"]
    assert messages[0]["role"] == "system"
    system_prompt = messages[0]["content"]
    assert system_prompt == build_grounding_system_prompt()
    # Key grounding rules must actually be present in the system prompt.
    for phrase in [
        "Answer ONLY using information contained in the supplied SOURCE context",
        "Do not invent",
        "does not establish an answer",
        "SOURCE_",
        "evidence_status",
        "partially_supported",
    ]:
        assert phrase in system_prompt

    assert messages[1]["role"] == "user"
    user_prompt = messages[1]["content"]
    assert user_prompt == build_context_prompt("Do appliances need certification?", SAMPLE_CONTEXT)
    assert "Do appliances need certification?" in user_prompt
    assert "[SOURCE_1]" in user_prompt
    assert "[SOURCE_2]" in user_prompt
    assert "BIS certification before sale" in user_prompt  # real clause text present
    assert "Earthing requirements" in user_prompt


# ---------------------------------------------------------------------------
# 5 & 6. Citation identifiers validated; unknown identifiers rejected
# ---------------------------------------------------------------------------

def test_valid_citation_ids_are_resolved_to_real_context_items():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"answer": "Answer citing [SOURCE_2].", "citation_ids": ["SOURCE_2"]})
    )
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is True
    assert result.cited_indices == [2]


def test_unknown_citation_identifiers_are_silently_dropped():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {"answer": "Answer.", "citation_ids": ["SOURCE_1", "SOURCE_99", "NOT_A_SOURCE", 123, None]}
        )
    )
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    # Only SOURCE_1 is real; everything else is dropped, not surfaced.
    assert result.cited_indices == [1]
    assert result.grounded is True


def test_citation_ids_outside_current_request_context_never_leak_into_response():
    """Even if the model 'remembers' a SOURCE id from a different request's
    context (e.g. SOURCE_5 when only SOURCE_1/2 were sent), it cannot be
    resolved because resolution only looks at the context actually given."""
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"answer": "Answer.", "citation_ids": ["SOURCE_5"]})
    )
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    # Stale SOURCE ids are rejected. A non-abstaining answer with no valid
    # citation is treated as an unsupported claim and falls back to extractive.
    expected = ExtractiveAnswerGenerationProvider().generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == expected.answer_text
    assert result.cited_indices == expected.cited_indices
    assert "SOURCE_5" not in result.answer_text


# ---------------------------------------------------------------------------
# 7. Fabricated citation metadata cannot enter the final response
# ---------------------------------------------------------------------------

def test_llm_cannot_fabricate_citation_metadata_fields():
    """The LLM JSON schema only carries `answer` and `citation_ids` (plain
    strings) — even if a malicious/buggy model stuffs extra fields into
    the JSON (e.g. a fake url/clause_number), those fields are never read
    or surfaced anywhere; GeneratedAnswer has no slot for them."""
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Answer. [SOURCE_1]",
                "citation_ids": ["SOURCE_1"],
                "source_url": "https://evil.example/fabricated.pdf",
                "clause_number": "99.99",
                "page_number": 9999,
                "standard_number": "IS 0000",
            }
        )
    )
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.cited_indices == [1]
    # GeneratedAnswer simply has no fields for fabricated metadata to
    # occupy — the extra JSON keys are ignored entirely.
    assert not hasattr(result, "source_url")
    assert not hasattr(result, "clause_number")
    assert "evil.example" not in result.answer_text


# ---------------------------------------------------------------------------
# 8. Malformed provider output handled (fallback)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_content",
    [
        "this is not json at all",
        "",
        None,
        json.dumps({"citation_ids": ["SOURCE_1"]}),  # missing "answer"
        json.dumps({"answer": "   "}),  # blank answer
        json.dumps(["not", "a", "dict"]),
        json.dumps(42),
    ],
)
def test_malformed_llm_output_falls_back_to_extractive(bad_content):
    fake_client = FakeOpenAIClient(response_content=bad_content)
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)

    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = fallback.generate("question", SAMPLE_CONTEXT)

    assert result.answer_text == expected.answer_text
    assert result.grounded == expected.grounded
    assert result.cited_indices == expected.cited_indices


def test_llm_output_wrapped_in_prose_is_still_salvaged():
    """Defensive parsing: some models wrap JSON in extra text despite
    instructions. A best-effort {...} extraction should still work."""
    wrapped = 'Sure, here is the JSON:\n{"answer": "Certification required. [SOURCE_1]", "citation_ids": ["SOURCE_1"]}\nHope that helps!'
    fake_client = FakeOpenAIClient(response_content=wrapped)
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is True
    assert result.cited_indices == [1]


# ---------------------------------------------------------------------------
# 9. Provider timeout/error handled (fallback)
# ---------------------------------------------------------------------------

def test_llm_network_error_falls_back_to_extractive():
    fake_client = FakeOpenAIClient(raise_exc=ConnectionError("simulated network failure"))
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)

    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = fallback.generate("question", SAMPLE_CONTEXT)

    assert result.answer_text == expected.answer_text
    assert result.grounded == expected.grounded


def test_llm_timeout_error_falls_back_to_extractive():
    class FakeTimeout(Exception):
        pass

    fake_client = FakeOpenAIClient(raise_exc=FakeTimeout("simulated timeout"))
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is True  # extractive fallback IS grounded (context exists)
    assert "certification" in result.answer_text.lower() or "explicitly supported" in result.answer_text.lower()


def test_llm_error_message_never_appears_in_returned_answer():
    """Exceptions must not leak into the user-facing answer text (no stack
    traces, no exception messages that might contain request details)."""
    fake_client = FakeOpenAIClient(raise_exc=RuntimeError("Authorization: Bearer sk-SECRETVALUE123"))
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert "SECRETVALUE123" not in result.answer_text
    assert "Authorization" not in result.answer_text
    assert "Bearer" not in result.answer_text


# ---------------------------------------------------------------------------
# 10. Missing API key behavior
# ---------------------------------------------------------------------------

def test_missing_api_key_falls_back_without_any_network_attempt():
    provider = LLMAnswerGenerationProvider(api_key=None)  # no client injected either

    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = ExtractiveAnswerGenerationProvider().generate("question", SAMPLE_CONTEXT)

    assert result.answer_text == expected.answer_text
    assert result.grounded == expected.grounded
    assert result.cited_indices == expected.cited_indices


def test_missing_api_key_with_empty_api_key_string_also_falls_back():
    provider = LLMAnswerGenerationProvider(api_key="")
    result = provider.generate("question", SAMPLE_CONTEXT)
    assert result.grounded is True  # extractive fallback grounds on real context


def test_empty_context_never_triggers_a_network_call_even_with_valid_key():
    fake_client = FakeOpenAIClient(response_content=json.dumps({"answer": "x", "citation_ids": []}))
    provider = _make_provider(fake_client)

    result = provider.generate("some question", [])

    assert fake_client.calls == []  # no call was made — cost control
    assert result.grounded is False
    assert "does not establish an answer" in result.answer_text


# ---------------------------------------------------------------------------
# get_default_answer_provider() — ANSWER_PROVIDER config selection
# ---------------------------------------------------------------------------

def test_default_provider_is_extractive_when_unconfigured(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ANSWER_PROVIDER", "extractive")
    provider = get_default_answer_provider()
    assert isinstance(provider, ExtractiveAnswerGenerationProvider)


def test_llm_provider_selected_but_no_key_behaves_safely(monkeypatch):
    """Selecting ANSWER_PROVIDER=llm without an API key must not crash the
    app or attempt a network call — every generate() call transparently
    falls back."""
    from app.config import settings

    monkeypatch.setattr(settings, "ANSWER_PROVIDER", "llm")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    provider = get_default_answer_provider()
    assert isinstance(provider, LLMAnswerGenerationProvider)

    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = ExtractiveAnswerGenerationProvider().generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == expected.answer_text


# ---------------------------------------------------------------------------
# 12. Fixed scope note always present on grounded LLM answers
# ---------------------------------------------------------------------------

def test_corpus_scope_note_appended_to_grounded_llm_answers():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"answer": "Yes, certification is required.", "citation_ids": ["SOURCE_1"]})
    )
    provider = _make_provider(fake_client)

    result = provider.generate("question", SAMPLE_CONTEXT)

    assert CORPUS_SCOPE_NOTE in result.answer_text
    assert "Yes, certification is required." in result.answer_text


# ---------------------------------------------------------------------------
# Milestone 10: evidence quality + adversarial citation safety
# ---------------------------------------------------------------------------

def test_fabricated_source_ids_are_rejected():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Penalty is five years. [SOURCE_99]",
                "citation_ids": ["SOURCE_99", "SOURCE_0", "SOURCE_999"],
                "evidence_status": "supported",
            }
        )
    )
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)

    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = fallback.generate("question", SAMPLE_CONTEXT)

    assert result.answer_text == expected.answer_text
    assert 99 not in result.cited_indices
    assert "SOURCE_99" not in result.answer_text


def test_malformed_citation_ids_are_rejected():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Certification is required. [SOURCE_1]",
                "citation_ids": ["SOURCE_1.5", "SOURCE_", "SOURCE_abc", "1", "[1]", "SOURCE 1"],
                "evidence_status": "supported",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    # Inline [SOURCE_1] is harvested; malformed citation_ids are dropped.
    assert result.cited_indices == [1]
    assert result.evidence_status == "supported"
    assert "[1]" in result.answer_text
    assert "SOURCE_1.5" not in result.answer_text


def test_duplicate_citation_ids_are_normalized():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Certification is required. [SOURCE_1] [SOURCE_1]",
                "citation_ids": ["SOURCE_1", "SOURCE_1", "[SOURCE_1]", "source_1"],
                "evidence_status": "supported",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.cited_indices == [1]
    assert result.answer_text.count("[1]") == 2
    assert "SOURCE_1" not in result.answer_text


def test_citations_outside_supplied_context_are_dropped():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Earthing applies. [SOURCE_2] and also [SOURCE_7]",
                "citation_ids": ["SOURCE_2", "SOURCE_7"],
                "evidence_status": "supported",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.cited_indices == [2]
    assert "[2]" in result.answer_text
    assert "SOURCE_7" not in result.answer_text
    assert "[7]" not in result.answer_text


def test_missing_answer_field_falls_back_to_extractive():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"citation_ids": ["SOURCE_1"], "evidence_status": "supported"})
    )
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)
    result = provider.generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == fallback.generate("question", SAMPLE_CONTEXT).answer_text


def test_malformed_json_falls_back_to_extractive():
    fake_client = FakeOpenAIClient(response_content="{answer: not json")
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)
    result = provider.generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == fallback.generate("question", SAMPLE_CONTEXT).answer_text


def test_unsupported_claims_without_valid_citations_fall_back():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "The penalty is imprisonment for five years and a fine of ten lakh rupees.",
                "citation_ids": [],
                "evidence_status": "supported",
            }
        )
    )
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)
    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = fallback.generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == expected.answer_text
    assert result.evidence_status == expected.evidence_status
    assert "ten lakh" not in result.answer_text


def test_empty_context_skips_llm_and_is_insufficient():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps({"answer": "hallucinated", "citation_ids": ["SOURCE_1"]})
    )
    provider = _make_provider(fake_client)
    result = provider.generate("some question", [])

    assert fake_client.calls == []
    assert result.grounded is False
    assert result.cited_indices == []
    assert result.evidence_status == "insufficient"
    assert "does not establish an answer" in result.answer_text


def test_multi_context_answer_synthesizes_and_cites_both_sources():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": (
                    "Manufacturers must obtain BIS certification before sale [SOURCE_1], "
                    "and household appliances must meet earthing requirements [SOURCE_2]."
                ),
                "citation_ids": ["SOURCE_1", "SOURCE_2"],
                "evidence_status": "supported",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate(
        "What certification and earthing rules apply to household appliances?",
        SAMPLE_CONTEXT,
    )

    assert result.grounded is True
    assert result.evidence_status == "supported"
    assert result.cited_indices == [1, 2]
    assert "[1]" in result.answer_text
    assert "[2]" in result.answer_text
    assert "SOURCE_1" not in result.answer_text
    assert "certification" in result.answer_text.lower()
    assert "earthing" in result.answer_text.lower()


def test_partially_supported_status_is_preserved_with_real_citations():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Certification before sale is required [SOURCE_1]. Earthing details for this product are not in the supplied material.",
                "citation_ids": ["SOURCE_1"],
                "evidence_status": "partially_supported",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is True
    assert result.evidence_status == "partially_supported"
    assert result.cited_indices == [1]


def test_llm_insufficient_status_clears_citations():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "The supplied BIS material does not establish an answer to this question.",
                "citation_ids": ["SOURCE_1"],
                "evidence_status": "insufficient",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is False
    assert result.evidence_status == "insufficient"
    assert result.cited_indices == []


def test_llm_out_of_scope_status_clears_citations():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "This question is outside the currently ingested BIS corpus.",
                "citation_ids": ["SOURCE_2"],
                "evidence_status": "out_of_scope",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.grounded is False
    assert result.evidence_status == "out_of_scope"
    assert result.cited_indices == []


def test_extractive_fallback_is_used_on_provider_error():
    fake_client = FakeOpenAIClient(raise_exc=RuntimeError("boom"))
    fallback = ExtractiveAnswerGenerationProvider()
    provider = _make_provider(fake_client, fallback=fallback)
    result = provider.generate("question", SAMPLE_CONTEXT)
    expected = fallback.generate("question", SAMPLE_CONTEXT)
    assert result.answer_text == expected.answer_text
    assert result.evidence_status == "supported"
    assert result.cited_indices == [1, 2]


def test_llm_cannot_control_clause_or_document_identity_fields():
    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {
                "answer": "Certification is required. [SOURCE_1]",
                "citation_ids": ["SOURCE_1"],
                "evidence_status": "supported",
                "clause_id": 9999,
                "clause_number": "99.99",
                "page_number": 9999,
                "standard_id": 42,
                "document_id": 42,
                "source_url": "https://evil.example/fabricated.pdf",
            }
        )
    )
    provider = _make_provider(fake_client)
    result = provider.generate("question", SAMPLE_CONTEXT)

    assert result.cited_indices == [1]
    assert not hasattr(result, "clause_id")
    assert not hasattr(result, "document_id")
    assert not hasattr(result, "source_url")
    assert "evil.example" not in result.answer_text
    assert "99.99" not in result.answer_text
