import { useMemo, useState } from "react";

import type { AnswerResponse, EvidenceStatus } from "../api/types";
import type { AssistantError } from "../api/types";

interface AnswerPanelProps {
  loading: boolean;
  error: AssistantError | null;
  result: AnswerResponse | null;
  question: string | null;
  onAskAnother: () => void;
}

const EVIDENCE_COPY: Record<
  EvidenceStatus,
  { kicker: string; heading: string; panelClass: string; abstained: boolean }
> = {
  supported: {
    kicker: "Supported by retrieved clauses",
    heading: "Answer",
    panelClass: "panel--grounded",
    abstained: false,
  },
  partially_supported: {
    kicker: "Partially supported",
    heading: "Partial answer",
    panelClass: "panel--partial",
    abstained: false,
  },
  insufficient: {
    kicker: "Insufficient evidence",
    heading: "No sufficiently relevant clause found",
    panelClass: "panel--insufficient",
    abstained: true,
  },
  out_of_scope: {
    kicker: "Outside the ingested corpus",
    heading: "Out of scope",
    panelClass: "panel--insufficient",
    abstained: true,
  },
};

function evidencePresentation(result: AnswerResponse) {
  if (result.evidence_status && EVIDENCE_COPY[result.evidence_status]) {
    return EVIDENCE_COPY[result.evidence_status];
  }
  return result.grounded ? EVIDENCE_COPY.supported : EVIDENCE_COPY.insufficient;
}

function AnswerBody({ text }: { text: string }) {
  const nodes = useMemo(() => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (!match) {
        return <span key={index}>{part}</span>;
      }
      const n = match[1];
      return (
        <a key={index} className="cite-chip" href={`#citation-${n}`}>
          [{n}]
        </a>
      );
    });
  }, [text]);

  return <div className="answer-body">{nodes}</div>;
}

export function AnswerPanel({ loading, error, result, question, onAskAnother }: AnswerPanelProps) {
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    if (!result) {
      return;
    }
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  if (loading) {
    return (
      <section className="panel" aria-busy="true" data-testid="loading-state">
        <p className="panel__kicker">Retrieving</p>
        <h2>Searching the ingested BIS corpus</h2>
        <p className="muted">Vector retrieval (API default), then a grounded answer with citations from real clause rows.</p>
        <div className="skeleton" />
        <div className="skeleton skeleton--short" />
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel panel--error" role="alert" data-testid="error-state">
        <p className="panel__kicker">Request failed{error.status ? ` · HTTP ${error.status}` : ""}</p>
        <h2>The assistant could not complete this question</h2>
        <p>{error.message}</p>
        <button type="button" className="btn-secondary" onClick={onAskAnother}>
          Ask another question
        </button>
      </section>
    );
  }

  if (!result) {
    return null;
  }

  const presentation = evidencePresentation(result);

  return (
    <section
      className={`panel ${presentation.panelClass}`}
      aria-live="polite"
      data-testid="answer-panel"
    >
      <div className="panel__meta">
        <p className="panel__kicker">{presentation.kicker}</p>
        <p className="badges">
          <span className="badge">method: {result.retrieval_method}</span>
          <span className="badge">context items: {result.context_used}</span>
          <span className="badge">citations: {result.citations.length}</span>
          <span className="badge">evidence: {result.evidence_status ?? (result.grounded ? "supported" : "insufficient")}</span>
        </p>
      </div>
      <h2>{presentation.heading}</h2>
      {question ? <p className="asked">Asked: {question}</p> : null}
      {result.evidence_status === "partially_supported" ? (
        <p className="callout" data-testid="partial-banner">
          Only some parts of this question are supported by the retrieved clauses. This label is categorical, not a numeric confidence score.
        </p>
      ) : null}
      {presentation.abstained ? (
        <p className="callout" data-testid="insufficient-banner">
          {result.evidence_status === "out_of_scope"
            ? "This question is outside the ingested BIS corpus. Absence of an answer here does not mean BIS has no such requirement elsewhere."
            : "The API returned insufficient evidence. Absence of an answer here does not mean BIS has no such requirement elsewhere."}
        </p>
      ) : null}
      <AnswerBody text={result.answer} />
      <div className="panel__actions">
        <button type="button" className="btn-secondary" onClick={copyAnswer} disabled={!result.answer}>
          {copied ? "Copied" : "Copy answer"}
        </button>
        <button type="button" className="btn-secondary" onClick={onAskAnother}>
          Ask another question
        </button>
      </div>
    </section>
  );
}
