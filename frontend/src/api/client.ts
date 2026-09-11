import type { AnswerResponse, AssistantError, AssistantErrorKind, HealthResponse } from "./types";
import { MAX_QUERY_LENGTH } from "./types";

export { MAX_QUERY_LENGTH };

function apiBase(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (configured === undefined || configured === "") {
    return "";
  }
  return configured.replace(/\/$/, "");
}

export function apiUrl(path: string): string {
  return `${apiBase()}${path}`;
}

function extractDetail(body: unknown): string | null {
  if (!body || typeof body !== "object") {
    return null;
  }
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && "msg" in item) {
          const msg = (item as { msg?: unknown }).msg;
          return typeof msg === "string" ? msg : null;
        }
        return null;
      })
      .filter((part): part is string => Boolean(part));
    if (parts.length > 0) {
      return parts.join("; ");
    }
  }
  return null;
}

export function errorFromResponse(status: number, body: unknown): AssistantError {
  const detail = extractDetail(body);
  let kind: AssistantErrorKind = "unknown";
  let message: string;

  if (status === 422) {
    kind = "validation";
    message =
      detail ??
      "The question could not be accepted. Check that it is not empty and is under 2,000 characters.";
  } else if (status === 400) {
    kind = "bad_request";
    message = detail ?? "The API rejected this request.";
  } else if (status === 503) {
    kind = "unavailable";
    message =
      detail ??
      "The backend is running but reported it is unavailable (database health check failed).";
  } else if (status >= 500) {
    kind = "server";
    message = detail ?? "The assistant encountered an internal server error.";
  } else {
    message = detail ?? `Request failed (HTTP ${status}).`;
  }

  return { kind, status, message };
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(apiUrl("/health"), {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw {
      kind: "network",
      status: null,
      message:
        "Could not reach the BIS assistant API. Start the FastAPI backend on http://127.0.0.1:8000.",
    } satisfies AssistantError;
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    throw errorFromResponse(response.status, body);
  }

  const parsed = body as HealthResponse;
  return {
    status: typeof parsed?.status === "string" ? parsed.status : "unknown",
    database: typeof parsed?.database === "string" ? parsed.database : "unknown",
  };
}

export async function askQuestion(query: string, signal?: AbortSignal): Promise<AnswerResponse> {
  let response: Response;
  try {
    response = await fetch(apiUrl("/api/search/answer"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw {
      kind: "network",
      status: null,
      message:
        "Could not reach the BIS assistant API. Confirm the backend is running on http://127.0.0.1:8000 and this page is served from the Vite dev server (proxy: /api → :8000).",
    } satisfies AssistantError;
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    throw errorFromResponse(response.status, body);
  }

  return body as AnswerResponse;
}
