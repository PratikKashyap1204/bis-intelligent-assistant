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
No hosted LLM (OpenAI/Anthropic/etc.) is wired into this project's
configuration (app/config.py only knows about DATABASE_URL and
EMBEDDING_MODEL_NAME — no LLM API key setting exists), and the brief for
this milestone explicitly says not to introduce one just for this work.
So the only generation strategy implemented here is a deterministic,
rule-based, *extractive* generator: it builds an answer strictly by
quoting/labelling the supplied context items, never adding a single fact
that isn't present in one of them.

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
  - It appends a fixed, honest scope note on every answer: this
    assistant's corpus is currently tiny (one Standard + one QCO
    circular), so it must never be read as covering all BIS standards.

Known limitation (documented, not hidden):
  This provider cannot judge *partial* sufficiency (e.g. "the retrieved
  clauses are topically related but don't fully answer this"). It only
  distinguishes "context found" vs "no context found". A future
  LLM-backed provider implementing the same protocol could reason about
  partial sufficiency — that is an explicit Milestone 4+ candidate, not
  attempted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

CORPUS_SCOPE_NOTE = (
    "Scope note: this assistant currently draws only on a small pilot BIS corpus "
    "(1 Indian Standard and 1 QCO circular, 129 clauses total). It does not cover "
    "all BIS standards, and absence of a requirement here does not mean BIS has no "
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
