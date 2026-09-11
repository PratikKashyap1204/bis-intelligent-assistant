"""
Answer generation abstraction (Milestone 3).

An ``AnswerGenerationProvider`` turns (question, retrieved context) into a
grounded answer. Nothing else in this codebase should hard-code a specific
generation strategy — always go through this interface (mirrors the
``EmbeddingProvider`` pattern from Milestone 2 in
app/services/embedding_provider.py), so a real LLM-backed provider can be
added later behind the same contract without touching RAGService or the
API layer.

Provider selected for this MVP: ExtractiveAnswerGenerationProvider
--------------------------------------------------------------------
The production default (``ANSWER_PROVIDER=extractive``) is a deterministic,
rule-based, *extractive* generator: it builds an answer strictly by
quoting/labelling the supplied context items, never adding a fact that
isn't present in one of them. An optional LLM provider exists (Milestone 4)
behind the same protocol and is used only when ``ANSWER_PROVIDER=llm``.

This is simultaneously:
  - the production default (nothing else is configured), and
  - the provider used by every automated test (fully offline, no network,
    no model weights, sub-millisecond) — satisfying the "mock provider
    for tests" requirement without maintaining two near-duplicate
    implementations of the same grounding logic.

Grounding guarantees (see generate() below):
  - If given no context, it returns a fixed answer stating the retrieved
    BIS material does not establish an answer — it never guesses.
  - If given context, every sentence in the answer is either a fixed
    template phrase or text copied (possibly truncated) from a context
    item's clause_text — nothing is synthesized from outside the context.
  - It always cites every context item it was given (RAGService is
    responsible for only handing it context that passed relevance/dedup/
    cap filtering — see app/services/rag.py), and citation indices are
    always a subset of the indices actually present in the input context,
    so a caller can safely map cited_indices back to real ContextItem
    objects without validating them "just in case" (RAGService still
    defensively re-validates — see RAGService._build_citations).
  - It appends a fixed, honest scope note on every answer describing the
    current pilot corpus (QCO/circular text plus standard metadata — see
    CORPUS_SCOPE_NOTE).

Known limitation (documented, not hidden):
  This provider cannot judge *partial* sufficiency (e.g. "the retrieved
  clauses are topically related but don't fully answer this"). It only
  distinguishes "context found" vs "no context found". A future
  LLM-backed provider implementing the same protocol could reason about
  partial sufficiency — that is an explicit Milestone 4+ candidate, not
  attempted here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

CORPUS_SCOPE_NOTE = (
    "Scope note: this assistant currently draws only on a small pilot BIS corpus "
    "(5 Indian Standard metadata records and 3 QCO circulars, 144 clauses). "
    "Ingested clause text comes from those QCO/circular documents, not from the "
    "full paid text of the Indian Standards themselves. It does not cover all "
    "BIS standards, and absence of a requirement here does not mean BIS has no "
    "such requirement elsewhere."
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "The available BIS material does not establish an answer to this question. "
    "No sufficiently relevant clause was found in the currently ingested BIS "
    "Standard/QCO content for this query."
)

# Clause text is truncated to this many characters per quoted excerpt so one
# very long clause can't dominate the whole answer.
_MAX_QUOTE_CHARS = 500


@dataclass
class ContextItem:
    """
    One piece of grounding context handed to an AnswerGenerationProvider.

    Built directly from a retrieval RetrievalResult (see
    app/services/rag.py) — never invented. ``index`` is the 1-based
    citation number a provider should use when referring to this item in
    its answer (e.g. "[2]").
    """

    index: int
    standard_number: Optional[str]
    standard_title: Optional[str]
    document_title: Optional[str]
    document_type: Optional[str]
    clause_number: Optional[str]
    clause_type: Optional[str]
    clause_text: str
    page_number: Optional[int]
    source_url: Optional[str]
    relevance_score: float
    retrieval_method: str
    # Stable identity (Clause.id). Optional so existing test fixtures
    # that omit it keep working; RAG always fills it from RetrievalResult.
    clause_id: Optional[int] = None
    document_id: Optional[int] = None


@dataclass
class GeneratedAnswer:
    """Result of AnswerGenerationProvider.generate()."""

    answer_text: str
    grounded: bool
    # Which ContextItem.index values this answer actually relies on.
    # RAGService trusts this list to build citations, but still discards
    # any index that isn't in the context it sent — see
    # RAGService._build_citations.
    cited_indices: List[int] = field(default_factory=list)


class AnswerGenerationProvider(Protocol):
    """Swappable answer-generation strategy (see module docstring)."""

    def generate(self, question: str, context: List[ContextItem]) -> GeneratedAnswer:
        ...


def _describe_source(item: ContextItem) -> str:
    """
    Human-readable label for a context item's source, preserving the
    Standard vs QCO/circular distinction (Milestone 3 grounding rule).
    """
    if item.document_type == "STANDARD" and item.standard_number:
        label = f"Indian Standard {item.standard_number}"
        if item.standard_title:
            label += f" ('{item.standard_title}')"
        return label

    if item.document_type == "QCO":
        label = f"QCO circular '{item.document_title}'" if item.document_title else "a QCO circular"
        if item.standard_number:
            label += f" (relating to {item.standard_number})"
        return label

    # Any other document type (AMENDMENT, SCHEME, FAQ, PRODUCT_MANUAL,
    # OTHER) or missing type — describe generically rather than guessing.
    if item.document_title:
        label = f"document '{item.document_title}'"
    elif item.standard_number:
        label = f"Indian Standard {item.standard_number}"
    else:
        label = "the retrieved BIS document"
    if item.document_type:
        label += f" ({item.document_type})"
    return label


def _quote(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _MAX_QUOTE_CHARS:
        text = text[:_MAX_QUOTE_CHARS].rstrip() + "…"
    return text


class ExtractiveAnswerGenerationProvider:
    """
    Deterministic, offline, extractive answer generator.

    See module docstring for the full grounding rationale. Behaviourally:
    given N context items, produces one line of intro, one bulleted block
    per item quoting its clause text with a "[i]" citation marker, and a
    fixed closing scope note. Given zero context items, returns the fixed
    INSUFFICIENT_CONTEXT_ANSWER instead (grounded=False).
    """

    def generate(self, question: str, context: List[ContextItem]) -> GeneratedAnswer:
        if not context:
            return GeneratedAnswer(
                answer_text=f"{INSUFFICIENT_CONTEXT_ANSWER}\n\n{CORPUS_SCOPE_NOTE}",
                grounded=False,
                cited_indices=[],
            )

        lines = [
            "Based only on the retrieved BIS material below, here is what is "
            "explicitly supported by the retrieved clauses:",
            "",
        ]
        for item in context:
            source = _describe_source(item)
            clause_label = f"Clause {item.clause_number}" if item.clause_number else "Clause"
            if item.clause_type:
                clause_label += f" ({item.clause_type})"
            page_part = f", page {item.page_number}" if item.page_number is not None else ""
            lines.append(
                f"[{item.index}] {source}, {clause_label}{page_part}: \"{_quote(item.clause_text)}\""
            )
        lines.append("")
        lines.append(
            "Anything not explicitly stated in the quoted clauses above (for example "
            "specific dates, penalties, or authorities not mentioned in them) cannot be "
            "determined from the currently retrieved BIS material."
        )
        lines.append("")
        lines.append(CORPUS_SCOPE_NOTE)

        return GeneratedAnswer(
            answer_text="\n".join(lines),
            grounded=True,
            cited_indices=[item.index for item in context],
        )


# ---------------------------------------------------------------------------
# LLM-backed provider (Milestone 4)
# ---------------------------------------------------------------------------
#
# Real vendor: OpenAI chat completions API (JSON-mode structured output).
# Selected because:
#   - .env.example already reserved OPENAI_API_KEY as the first placeholder
#     credential slot for this project (added, unused, back in an earlier
#     stage) — this is the vendor the project already anticipated.
#   - Its "json_object" response format gives reasonably reliable
#     structured output ({"answer": ..., "citation_ids": [...]}) without
#     requiring a heavier structured-outputs/function-calling setup.
#   - Low-cost small models (e.g. gpt-4o-mini) are inexpensive enough for
#     a development project used only via an explicit manual demo script.
#
# This class NEVER makes a network call unless:
#   1. it is actually selected (ANSWER_PROVIDER=llm, see get_default_answer_provider), AND
#   2. an API key is configured, AND
#   3. RAGService actually found non-empty context for the question.
# Any of those being false is handled locally (fallback / fixed message),
# with zero network activity.
#
# Citation safety (critical): the LLM's JSON output is NEVER used to build
# citation metadata directly. Its "citation_ids" are just strings like
# "SOURCE_2" that get validated against the real ContextItem indices that
# were actually sent (see _resolve_citation_ids) — anything that doesn't
# match a real, given SOURCE id is silently dropped. The LLM cannot cause
# a clause number, page number, standard number, or URL to appear in the
# final response; those always come from RAGService._build_citations,
# which only ever reads from real ContextItem/RetrievalResult objects
# (see app/services/rag.py). The LLM only ever contributes free-form
# answer text and a list of which of the given sources it used.


class LLMProviderUnavailableError(RuntimeError):
    """Raised internally when the LLM client cannot be used (e.g. no API key)."""


def build_grounding_system_prompt() -> str:
    """
    The fixed system prompt sent with every LLM call. Kept as a plain
    function (not a class) so it is trivially unit-testable and reviewable
    on its own — this is the one piece of text responsible for the
    grounding guarantees required of the LLM-backed provider.
    """
    return (
        "You are a grounded question-answering assistant for the Bureau of Indian "
        "Standards (BIS) domain. You are given a user question and a numbered list "
        "of SOURCE context items retrieved from a database of ingested BIS Standards "
        "and QCO (Quality Control Order) circulars.\n\n"
        "Follow these rules exactly:\n"
        "1. Answer ONLY using information contained in the supplied SOURCE context. "
        "Do not use outside knowledge, training data, or general assumptions about "
        "BIS, Indian law, or certification processes to fill in missing information.\n"
        "2. Do not invent or assume any clause, requirement, date, authority, penalty, "
        "standard number, or certification rule that is not explicitly present in the "
        "supplied context.\n"
        "3. If the supplied context does not establish an answer to the question, say "
        "so explicitly (e.g. 'The supplied BIS material does not establish an answer "
        "to this question.'). Do not guess.\n"
        "4. Clearly distinguish between a BIS Standard, a QCO (Quality Control Order), "
        "and a circular/other document type — use the type given for each SOURCE, "
        "never assume one.\n"
        "5. When you rely on a SOURCE, cite it using its exact identifier in square "
        "brackets, e.g. [SOURCE_2]. Only cite SOURCE identifiers that were given to "
        "you. Never invent new identifiers, and never cite a SOURCE for information "
        "it does not contain.\n"
        "6. Do not state or generate any citation metadata yourself (no clause "
        "numbers, page numbers, standard numbers, or URLs beyond what appears inside "
        "the SOURCE text) — the calling system attaches that metadata separately, "
        "keyed only by the SOURCE identifiers you cite.\n"
        "7. Respond with ONLY a single JSON object, no other text, of exactly this "
        "form:\n"
        '   {"answer": "<answer text, with inline [SOURCE_n] citations>", '
        '"citation_ids": ["SOURCE_n", ...]}\n'
        "   citation_ids must list every SOURCE identifier your answer actually "
        "relies on, and nothing else."
    )


def build_context_prompt(question: str, context: List[ContextItem]) -> str:
    """The per-request user message: the question plus labelled SOURCE_n blocks."""
    lines = [f"USER QUESTION:\n{question}\n", "RETRIEVED SOURCES:"]
    for item in context:
        source = _describe_source(item)
        lines.append(
            f"[SOURCE_{item.index}]\n"
            f"Type: {item.document_type or 'UNKNOWN'}\n"
            f"Source: {source}\n"
            f"Clause: {item.clause_number or 'N/A'} ({item.clause_type or 'N/A'})"
            f"{f', page {item.page_number}' if item.page_number is not None else ''}\n"
            f"Text: \"{_quote(item.clause_text)}\"\n"
        )
    return "\n".join(lines)


def _resolve_citation_ids(citation_ids: Any, context: List[ContextItem]) -> List[int]:
    """
    Map LLM-provided "SOURCE_n" strings back to real ContextItem indices.
    Anything that isn't a string, or doesn't match a real SOURCE id that
    was actually sent in this request, is silently dropped — this is the
    only path by which the LLM's output can influence citations at all.
    """
    if not isinstance(citation_ids, list):
        return []
    valid_ids = {f"SOURCE_{item.index}": item.index for item in context}
    resolved: List[int] = []
    for raw_id in citation_ids:
        if not isinstance(raw_id, str):
            continue
        idx = valid_ids.get(raw_id.strip())
        if idx is not None and idx not in resolved:
            resolved.append(idx)
    return resolved


def _parse_llm_json_response(
    raw_text: Optional[str], context: List[ContextItem]
) -> Optional[Tuple[str, List[int]]]:
    """
    Parse + validate the LLM's JSON output. Returns None (triggering a
    fallback to the deterministic provider) if the output is missing,
    not valid JSON, not an object, or missing a usable ``answer`` string.
    A missing/invalid ``citation_ids`` is tolerated (treated as empty) —
    that's a valid "no grounded citation" case, not malformed output.
    """
    if not raw_text or not raw_text.strip():
        return None

    data = None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Defensive: some models wrap JSON in prose/code fences despite
        # instructions. Try to salvage the first {...} block.
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    answer_text = data.get("answer")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None

    resolved_indices = _resolve_citation_ids(data.get("citation_ids"), context)
    return answer_text.strip(), resolved_indices


class LLMAnswerGenerationProvider:
    """
    Real LLM-backed AnswerGenerationProvider (OpenAI chat completions,
    JSON-mode). See the module-level comment block above for the full
    rationale, cost-control, and citation-safety design.

    Configuration is passed in explicitly (model/temperature/max tokens/
    timeout/api key) rather than read from app.config directly, so this
    class stays framework-agnostic and easily testable — wiring from
    environment variables happens once, in get_default_answer_provider().

    ``client`` is an optional dependency-injection point for tests: pass
    any object exposing ``.chat.completions.create(...)`` returning an
    object shaped like the OpenAI SDK's response (``.choices[0].message.
    content``). When not supplied, a real ``openai.OpenAI`` client is
    constructed lazily on first use (never at import time, and never if
    ``api_key`` is falsy).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 600,
        timeout: float = 20.0,
        client: Optional[Any] = None,
        fallback: Optional[AnswerGenerationProvider] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self._client = client
        self.fallback: AnswerGenerationProvider = fallback or ExtractiveAnswerGenerationProvider()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMProviderUnavailableError(
                "LLMAnswerGenerationProvider has no API key configured."
            )
        # Imported here (not at module level) so importing this module —
        # and using ExtractiveAnswerGenerationProvider or any other
        # provider — never requires the `openai` package to be installed
        # unless this class is actually used with a real client.
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def generate(self, question: str, context: List[ContextItem]) -> GeneratedAnswer:
        if not context:
            # No relevant context was found — RAGService already decided
            # this; there is nothing for an LLM to ground an answer in,
            # so skip the network call entirely (cost control + zero
            # hallucination risk) and return the same fixed message the
            # extractive provider would give for empty context.
            return self.fallback.generate(question, context)

        try:
            client = self._ensure_client()
        except LLMProviderUnavailableError:
            return self.fallback.generate(question, context)

        system_prompt = build_grounding_system_prompt()
        user_prompt = build_context_prompt(question, context)

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                timeout=self.timeout,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content
        except Exception:
            # Any network/timeout/auth/rate-limit/SDK error — fail
            # gracefully rather than propagate (which would surface a raw
            # provider error, and could risk leaking request details in a
            # traceback). Never fabricate an answer on error.
            return self.fallback.generate(question, context)

        parsed = _parse_llm_json_response(raw_text, context)
        if parsed is None:
            # Malformed output (not JSON, wrong shape, no usable answer).
            return self.fallback.generate(question, context)

        answer_text, cited_indices = parsed
        return GeneratedAnswer(
            answer_text=f"{answer_text}\n\n{CORPUS_SCOPE_NOTE}",
            grounded=len(cited_indices) > 0,
            cited_indices=cited_indices,
        )


def get_default_answer_provider() -> AnswerGenerationProvider:
    """
    Build the AnswerGenerationProvider selected by configuration
    (``ANSWER_PROVIDER`` in .env / app.config.settings).

    ``ANSWER_PROVIDER=extractive`` (default) -> ExtractiveAnswerGenerationProvider,
    fully offline, no key needed.
    ``ANSWER_PROVIDER=llm`` -> LLMAnswerGenerationProvider. If no API key is
    configured even when "llm" is selected, LLMAnswerGenerationProvider
    itself falls back to the extractive provider on every call (see
    generate() above) — so this never crashes or requires a key just to
    start the application.

    Imported lazily inside the function body (like
    get_default_embedding_provider in embedding_provider.py) to keep
    importing this module free of any app.config/env dependency for
    callers that only need a specific provider class directly.
    """
    from app.config import settings

    if settings.ANSWER_PROVIDER == "llm":
        return LLMAnswerGenerationProvider(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    return ExtractiveAnswerGenerationProvider()
