/**
 * Types matching backend/app/schemas/rag.py and GET /health.
 * Field names follow the API exactly — do not invent extra metadata.
 */

export const MAX_QUERY_LENGTH = 2000;

export interface Citation {
  index: number;
  standard_number: string | null;
  standard_title: string | null;
  document_title: string | null;
  document_type: string | null;
  clause_number: string | null;
  clause_type: string | null;
  page_number: number | null;
  source_url: string | null;
  relevance_score: number;
  clause_id: number | null;
  document_id: number | null;
}

export interface Source {
  standard_number: string | null;
  standard_title: string | null;
  document_title: string | null;
  document_type: string | null;
  clause_number: string | null;
  clause_type: string | null;
  clause_text: string | null;
  page_number: number | null;
  source_url: string | null;
  relevance_score: number;
  relevance_method: string;
  clause_id: number | null;
  document_id: number | null;
}

export type EvidenceStatus =
  | "supported"
  | "partially_supported"
  | "insufficient"
  | "out_of_scope";

export interface AnswerResponse {
  answer: string;
  citations: Citation[];
  retrieval_method: string;
  sources: Source[];
  grounded: boolean;
  context_used: number;
  evidence_status: EvidenceStatus;
}

export interface AnswerRequest {
  query: string;
}

export interface HealthResponse {
  status: string;
  database: string;
}

export type AssistantErrorKind =
  | "network"
  | "validation"
  | "bad_request"
  | "server"
  | "unavailable"
  | "unknown";

export interface AssistantError {
  kind: AssistantErrorKind;
  status: number | null;
  message: string;
}
