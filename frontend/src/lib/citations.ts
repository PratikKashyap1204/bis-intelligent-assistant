import type { Citation, Source } from "../api/types";

/** Join a citation to the matching retrieval source without inventing fields. */
export function sourceForCitation(citation: Citation, sources: Source[]): Source | undefined {
  if (citation.clause_id != null) {
    return sources.find((source) => source.clause_id === citation.clause_id);
  }
  return sources[citation.index - 1];
}

export function citationHeading(citation: Citation): string {
  if (citation.document_title) {
    return citation.document_title;
  }
  if (citation.standard_number && citation.standard_title) {
    return `${citation.standard_number} — ${citation.standard_title}`;
  }
  if (citation.standard_number) {
    return citation.standard_number;
  }
  return `Source [${citation.index}]`;
}
