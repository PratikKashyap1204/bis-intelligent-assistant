import type { FormEvent, KeyboardEvent } from "react";

import { MAX_QUERY_LENGTH } from "../api/types";

interface QuestionFormProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  canSubmit: boolean;
}

export function QuestionForm({
  query,
  onQueryChange,
  onSubmit,
  loading,
  canSubmit,
}: QuestionFormProps) {
  const remaining = MAX_QUERY_LENGTH - query.length;

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (canSubmit) {
      onSubmit();
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (canSubmit) {
        onSubmit();
      }
    }
  }

  const tooLong = query.length > MAX_QUERY_LENGTH;

  return (
    <form className="ask-form" onSubmit={handleSubmit} aria-describedby="ask-help">
      <label htmlFor="question">Question</label>
      <p id="ask-help" className="hint">
        Answers are retrieved from ingested QCO circulars and Indian Standard metadata only.
        Enter submits; Shift+Enter adds a new line.
      </p>
      <textarea
        id="question"
        name="query"
        rows={4}
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Example: Which appliances require BIS certification before sale?"
        maxLength={MAX_QUERY_LENGTH + 50}
        disabled={loading}
        aria-invalid={tooLong}
        aria-busy={loading}
      />
      <div className="ask-form__row">
        <p className={`counter${tooLong ? " is-bad" : ""}`}>
          {tooLong ? "Too long for the API (max 2,000 characters)." : `${remaining} characters remaining`}
        </p>
        <button type="submit" className="btn-primary" disabled={!canSubmit}>
          {loading ? "Retrieving…" : "Ask"}
        </button>
      </div>
    </form>
  );
}
