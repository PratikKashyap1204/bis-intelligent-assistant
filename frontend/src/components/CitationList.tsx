import type { Citation, Source } from "../api/types";
import { citationHeading, sourceForCitation } from "../lib/citations";

interface CitationListProps {
  citations: Citation[];
  sources: Source[];
  grounded: boolean;
}

function ProvenanceRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  return (
    <div className="prov">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function CitationCard({ citation, source }: { citation: Citation; source: Source | undefined }) {
  const heading = citationHeading(citation);
  const clauseText = source?.clause_text ?? null;
  const method = source?.relevance_method ?? null;

  return (
    <article className="citation-card" id={`citation-${citation.index}`} data-testid="citation-card">
      <details>
        <summary>
          <span className="citation-index">[{citation.index}]</span>
          <span className="citation-summary">
            <strong>{heading}</strong>
            <span className="citation-line">
              {[citation.document_type, citation.standard_number, citation.clause_number ? `clause ${citation.clause_number}` : null]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </span>
        </summary>
        <dl className="provenance">
          <ProvenanceRow label="Standard number" value={citation.standard_number} />
          <ProvenanceRow label="Standard title" value={citation.standard_title} />
          <ProvenanceRow label="Document title" value={citation.document_title} />
          <ProvenanceRow label="Document type" value={citation.document_type} />
          <ProvenanceRow label="Clause number" value={citation.clause_number} />
          <ProvenanceRow label="Clause type" value={citation.clause_type} />
          <ProvenanceRow label="Clause ID" value={citation.clause_id} />
          <ProvenanceRow label="Document ID" value={citation.document_id} />
          <ProvenanceRow label="Page" value={citation.page_number} />
          {citation.source_url ? (
            <div className="prov">
              <dt>Source URL</dt>
              <dd>
                <a href={citation.source_url} target="_blank" rel="noreferrer">
                  {citation.source_url}
                </a>
              </dd>
            </div>
          ) : null}
          {method ? <ProvenanceRow label="Retrieval method" value={method} /> : null}
          <ProvenanceRow
            label="Retrieval score"
            value={`${citation.relevance_score} (ranking score from the API, not answer confidence)`}
          />
        </dl>
        {clauseText ? (
          <blockquote className="clause-quote">
            <p>{clauseText}</p>
          </blockquote>
        ) : null}
      </details>
    </article>
  );
}

export function CitationList({ citations, sources, grounded }: CitationListProps) {
  return (
    <section className="sources" aria-label="Citations from the API response" data-testid="citation-list">
      <div className="sources__head">
        <h2>Sources</h2>
        <p>Only citations returned by <code>POST /api/search/answer</code> are shown.</p>
      </div>
      {citations.length === 0 ? (
        <p className="muted" data-testid="no-citations">
          {grounded
            ? "The response was marked grounded but included no citation objects."
            : "No citations — the API did not find sufficiently relevant clauses."}
        </p>
      ) : (
        citations.map((citation) => (
          <CitationCard
            key={`${citation.index}-${citation.clause_id ?? "none"}`}
            citation={citation}
            source={sourceForCitation(citation, sources)}
          />
        ))
      )}
    </section>
  );
}
