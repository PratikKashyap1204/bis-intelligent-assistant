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
  The extractive provider cannot judge *partial* sufficiency on its own
  (e.g. "the retrieved clauses are topically related but don't fully
  answer this"). It distinguishes empty context (insufficient / out of
  scope, the latter decided by RAGService's corpus-token gate) from
  non-empty context (supported, quoting every supplied item). The LLM
  provider (same protocol) may label ``partially_supported`` when the
  supplied sources only cover some asked clauses — still without a
  numeric confidence score.
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

OUT_OF_SCOPE_ANSWER = (
    "This question is outside the currently ingested BIS corpus. "
    "The available BIS material does not establish an answer to this question "
    "because the ingested Standard/QCO content does not cover this topic."
)

# Categorical evidence labels (Milestone 10). These are not a calibrated
# numeric confidence score — they only describe how the supplied context
# relates to the question. Do not invent a 0–1 confidence from them.
EVIDENCE_SUPPORTED = "supported"
EVIDENCE_PARTIALLY_SUPPORTED = "partially_supported"
EVIDENCE_INSUFFICIENT = "insufficient"
EVIDENCE_OUT_OF_SCOPE = "out_of_scope"
VALID_EVIDENCE_STATUSES = frozenset(
    {
        EVIDENCE_SUPPORTED,
        EVIDENCE_PARTIALLY_SUPPORTED,
        EVIDENCE_INSUFFICIENT,
        EVIDENCE_OUT_OF_SCOPE,
    }
)
_SUPPORTED_EVIDENCE_STATUSES = frozenset(
    {EVIDENCE_SUPPORTED, EVIDENCE_PARTIALLY_SUPPORTED}
)

_EVIDENCE_STATUS_ALIASES = {
    "supported": EVIDENCE_SUPPORTED,
    "fully_supported": EVIDENCE_SUPPORTED,
    "partial": EVIDENCE_PARTIALLY_SUPPORTED,
    "partially_supported": EVIDENCE_PARTIALLY_SUPPORTED,
    "insufficient": EVIDENCE_INSUFFICIENT,
    "insufficient_evidence": EVIDENCE_INSUFFICIENT,
    "not_enough_information": EVIDENCE_INSUFFICIENT,
    "out_of_scope": EVIDENCE_OUT_OF_SCOPE,
    "outofscope": EVIDENCE_OUT_OF_SCOPE,
    "outside_corpus": EVIDENCE_OUT_OF_SCOPE,
}

_ABSTENTION_HINTS = (
    "does not establish",
    "not enough information",
    "insufficient evidence",
    "cannot be determined",
    "outside the currently ingested",
    "does not cover this topic",
    "no sufficiently relevant clause",
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
    # Categorical evidence label. None means "provider did not decide"
    # (legacy stubs); RAGService then infers from grounded/empty context.
    # Never a numeric confidence score.
    evidence_status: Optional[str] = None


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


def _first_sentence(text: str) -> str:
    """Deterministic first-sentence excerpt for extractive multi-clause leads."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    sentence = parts[0].strip()
    if len(sentence) > 280:
        sentence = sentence[:277].rstrip() + "…"
    return sentence


def looks_like_abstention(text: str) -> bool:
    """True when answer text already says the sources do not answer the question."""
    lowered = (text or "").lower()
    return any(hint in lowered for hint in _ABSTENTION_HINTS)


def normalize_evidence_status(raw: Any) -> Optional[str]:
    """Map a provider/LLM evidence label onto the closed Milestone 10 set."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _EVIDENCE_STATUS_ALIASES.get(key)


def build_abstention_answer(evidence_status: str) -> str:
    body = (
        OUT_OF_SCOPE_ANSWER
        if evidence_status == EVIDENCE_OUT_OF_SCOPE
        else INSUFFICIENT_CONTEXT_ANSWER
    )
    return f"{body}\n\n{CORPUS_SCOPE_NOTE}"


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
                answer_text=build_abstention_answer(EVIDENCE_INSUFFICIENT),
                grounded=False,
                cited_indices=[],
                evidence_status=EVIDENCE_INSUFFICIENT,
            )

        lines = [
            "Based only on the retrieved BIS material below, here is what is "
            "explicitly supported by the retrieved clauses:",
            "",
        ]
        if len(context) > 1:
            lines.append("The retrieved clauses together support the following points:")
            for item in context:
                sentence = _first_sentence(item.clause_text)
                if sentence:
                    lines.append(f"{sentence} [{item.index}]")
            lines.append("")
            lines.append("Quoted source text:")
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
            evidence_status=EVIDENCE_SUPPORTED,
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
#     structured output ({"answer": ..., "citation_ids": [...],
#     "evidence_status": ...}) without requiring a heavier
#     structured-outputs/function-calling setup.
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
        "3. If the question has several parts, you MAY synthesize a single coherent "
        "answer from multiple SOURCE items, but every substantive claim must be "
        "supported by at least one supplied SOURCE. Do not merge sources into a "
        "claim that none of them actually states.\n"
        "4. Choose exactly one evidence_status:\n"
        "   - \"supported\": the supplied sources fully establish the answer.\n"
        "   - \"partially_supported\": the sources establish some asked parts but not all.\n"
        "   - \"insufficient\": the sources are related or empty of a usable answer, "
        "but the question is still inside the BIS domain of this corpus.\n"
        "   - \"out_of_scope\": the question is outside what the supplied sources "
        "(and this ingested corpus) can address.\n"
        "   Do not invent a numeric confidence score.\n"
        "5. If evidence_status is insufficient or out_of_scope, do not present "
        "unsupported claims as facts. Say that the supplied material does not "
        "establish an answer. citation_ids must be empty in that case.\n"
        "6. Clearly distinguish between a BIS Standard, a QCO (Quality Control Order), "
        "and a circular/other document type — use the type given for each SOURCE, "
        "never assume one.\n"
        "7. When you rely on a SOURCE, cite it using its exact identifier in square "
        "brackets, e.g. [SOURCE_2]. Only cite SOURCE identifiers that were given to "
        "you in this request. Never invent new identifiers, never reuse identifiers "
        "from earlier questions, and never cite a SOURCE for information it does "
        "not contain.\n"
        "8. Do not state or generate any citation metadata yourself (no clause "
        "numbers, page numbers, standard numbers, clause_id, document_id, or URLs "
        "beyond what appears inside the SOURCE text) — the calling system attaches "
        "that metadata separately, keyed only by the SOURCE identifiers you cite.\n"
        "9. Respond with ONLY a single JSON object, no other text, of exactly this "
        "form:\n"
        '   {"answer": "<answer text, with inline [SOURCE_n] citations>", '
        '"citation_ids": ["SOURCE_n", ...], '
        '"evidence_status": "supported|partially_supported|insufficient|out_of_scope"}\n'
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


_SOURCE_TOKEN_RE = re.compile(r"^\[?\s*SOURCE_(\d+)\s*\]?$", re.IGNORECASE)
_INLINE_SOURCE_RE = re.compile(r"\[SOURCE_(\d+)\]", re.IGNORECASE)
_NUMERIC_CITE_RE = re.compile(r"\[(\d+)\]")


def _canonical_source_index(raw: Any) -> Optional[int]:
    """Return n for a well-formed SOURCE_n token, else None."""
    if not isinstance(raw, str):
        return None
    match = _SOURCE_TOKEN_RE.match(raw.strip())
    if not match:
        return None
    return int(match.group(1))


def _resolve_citation_ids(citation_ids: Any, context: List[ContextItem]) -> List[int]:
    """
    Map LLM-provided "SOURCE_n" strings back to real ContextItem indices.
    Anything that isn't a well-formed SOURCE id that was actually sent in
    this request is silently dropped — this is the only path by which the
    LLM's output can influence citations at all.

    Accepts SOURCE_n / source_n / [SOURCE_n]. Rejects malformed tokens
    (SOURCE_1.5, SOURCE_, SOURCE_abc), non-strings, duplicates, and ids
    that were not supplied in ``context``.
    """
    if not isinstance(citation_ids, list):
        return []
    valid_indices = {item.index for item in context}
    resolved: List[int] = []
    for raw_id in citation_ids:
        idx = _canonical_source_index(raw_id)
        if idx is None or idx not in valid_indices or idx in resolved:
            continue
        resolved.append(idx)
    return resolved


def _harvest_inline_source_indices(answer_text: str, context: List[ContextItem]) -> List[int]:
    valid_indices = {item.index for item in context}
    harvested: List[int] = []
    for match in _INLINE_SOURCE_RE.finditer(answer_text or ""):
        idx = int(match.group(1))
        if idx in valid_indices and idx not in harvested:
            harvested.append(idx)
    return harvested


def merge_cited_indices(*groups: List[int]) -> List[int]:
    merged: List[int] = []
    for group in groups:
        for idx in group:
            if idx not in merged:
                merged.append(idx)
    return merged


def rewrite_answer_citations(answer_text: str, valid_indices: List[int]) -> str:
    """
    Normalize inline citations to ``[n]`` using only indices present in
    the supplied context. Unknown/malformed SOURCE markers and numeric
    citation chips that do not match a real context item are stripped.
    """
    allowed = set(valid_indices)

    def _source_repl(match: re.Match) -> str:
        idx = int(match.group(1))
        return f"[{idx}]" if idx in allowed else ""

    rewritten = _INLINE_SOURCE_RE.sub(_source_repl, answer_text or "")

    def _numeric_repl(match: re.Match) -> str:
        idx = int(match.group(1))
        return match.group(0) if idx in allowed else ""

    rewritten = _NUMERIC_CITE_RE.sub(_numeric_repl, rewritten)
    rewritten = re.sub(r"[ \t]+\n", "\n", rewritten)
    rewritten = re.sub(r" {2,}", " ", rewritten)
    return rewritten.strip()


def _parse_llm_json_response(
    raw_text: Optional[str], context: List[ContextItem]
) -> Optional[Tuple[str, List[int], str]]:
    """
    Parse + validate the LLM's JSON output. Returns None (triggering a
    fallback to the deterministic provider) if the output is missing,
    not valid JSON, not an object, or missing a usable ``answer`` string.
    A missing/invalid ``citation_ids`` is tolerated (treated as empty).
    Unknown evidence_status values are ignored (inferred from citations).

    Also returns None when the model claims support (or makes a
    non-abstaining answer) without any valid citation into the supplied
    context — that is treated as an unsupported-claim failure, not as a
    grounded answer.
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

    from_ids = _resolve_citation_ids(data.get("citation_ids"), context)
    from_inline = _harvest_inline_source_indices(answer_text, context)
    cited_indices = merge_cited_indices(from_ids, from_inline)
    evidence_status = normalize_evidence_status(data.get("evidence_status"))

    abstaining = looks_like_abstention(answer_text)
    if evidence_status in _SUPPORTED_EVIDENCE_STATUSES and not cited_indices:
        return None
    if evidence_status is None:
        if cited_indices:
            evidence_status = EVIDENCE_SUPPORTED
        elif abstaining:
            evidence_status = EVIDENCE_INSUFFICIENT
        else:
            # Positive-sounding answer with no valid SOURCE ids.
            return None
    if evidence_status in (EVIDENCE_INSUFFICIENT, EVIDENCE_OUT_OF_SCOPE):
        cited_indices = []

    rewritten = rewrite_answer_citations(answer_text.strip(), cited_indices)
    if not rewritten:
        return None
    return rewritten, cited_indices, evidence_status


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
            # Malformed output (not JSON, wrong shape, no usable answer)
            # or an unsupported claim with no valid SOURCE ids.
            return self.fallback.generate(question, context)

        answer_text, cited_indices, evidence_status = parsed
        grounded = evidence_status in _SUPPORTED_EVIDENCE_STATUSES and len(cited_indices) > 0
        return GeneratedAnswer(
            answer_text=f"{answer_text}\n\n{CORPUS_SCOPE_NOTE}",
            grounded=grounded,
            cited_indices=cited_indices,
            evidence_status=evidence_status,
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
