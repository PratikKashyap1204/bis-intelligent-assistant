import { AnswerPanel } from "./components/AnswerPanel";
import { CitationList } from "./components/CitationList";
import { EmptyState } from "./components/EmptyState";
import { Header } from "./components/Header";
import { QuestionForm } from "./components/QuestionForm";
import { SessionHistory } from "./components/SessionHistory";
import { useAssistant } from "./hooks/useAssistant";
import { useHealth } from "./hooks/useHealth";

export default function App() {
  const connection = useHealth();
  const assistant = useAssistant();

  const showEmpty = assistant.status === "idle" && !assistant.result && !assistant.error;
  const showSources = assistant.status === "success" && assistant.result !== null;

  return (
    <div className="page">
      <a className="skip-link" href="#question">
        Skip to question
      </a>
      <Header connection={connection} />

      <main>
        <p className="lede">
          Ask a question about BIS Quality Control Orders and related Indian Standard metadata,
          then inspect exactly which retrieved clauses the answer was grounded in.
        </p>

        <aside className="scope-note" aria-label="Corpus scope">
          <strong>Grounded in the ingested pilot corpus.</strong> 5 Indian Standard metadata
          records, 3 QCO circulars, 144 clauses. Clause text is from those circulars, not the
          paid full text of Indian Standards. Absence of a requirement here does not mean BIS
          has no such requirement elsewhere.
        </aside>

        {connection.kind === "unreachable" ? (
          <p className="banner-warn" role="status">
            {connection.message}
          </p>
        ) : null}

        <QuestionForm
          query={assistant.query}
          onQueryChange={assistant.setQuery}
          onSubmit={() => {
            void assistant.submit();
          }}
          loading={assistant.status === "loading"}
          canSubmit={assistant.canSubmit}
        />

        <SessionHistory
          items={assistant.history}
          onSelect={assistant.replay}
          disabled={assistant.status === "loading"}
        />

        <div className="workspace">
          <div className="workspace__answer">
            {showEmpty ? <EmptyState /> : null}
            <AnswerPanel
              loading={assistant.status === "loading"}
              error={assistant.error}
              result={assistant.result}
              question={assistant.lastQuestion}
              onAskAnother={assistant.askAnother}
            />
          </div>
          {showSources && assistant.result ? (
            <CitationList
              citations={assistant.result.citations}
              sources={assistant.result.sources}
              grounded={assistant.result.grounded}
            />
          ) : null}
        </div>
      </main>

      <footer className="site-footer">
        <p>
          SIH pilot · FastAPI <code>POST /api/search/answer</code> · retrieval default is vector ·
          citations are API provenance, not generated metadata.
        </p>
      </footer>
    </div>
  );
}
