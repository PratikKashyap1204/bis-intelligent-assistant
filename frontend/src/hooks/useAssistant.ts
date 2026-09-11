import { useCallback, useMemo, useState } from "react";

import { askQuestion } from "../api/client";
import type { AnswerResponse, AssistantError } from "../api/types";
import { MAX_QUERY_LENGTH } from "../api/types";
import { loadHistory, pushHistory, type HistoryItem } from "../lib/history";

export type AssistantStatus = "idle" | "loading" | "success" | "error";

function asAssistantError(error: unknown): AssistantError {
  if (error && typeof error === "object" && "kind" in error && "message" in error) {
    return error as AssistantError;
  }
  return {
    kind: "unknown",
    status: null,
    message: "An unexpected error occurred while asking the assistant.",
  };
}

export function useAssistant() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<AssistantError | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory());

  const trimmed = query.trim();
  const canSubmit = trimmed.length > 0 && trimmed.length <= MAX_QUERY_LENGTH && status !== "loading";

  const submit = useCallback(
    async (override?: string) => {
      const nextQuery = (override ?? query).trim();
      if (!nextQuery || nextQuery.length > MAX_QUERY_LENGTH) {
        return;
      }
      setQuery(nextQuery);
      setStatus("loading");
      setError(null);
      setResult(null);
      setLastQuestion(nextQuery);
      try {
        const response = await askQuestion(nextQuery);
        setResult(response);
        setStatus("success");
        setHistory(pushHistory(nextQuery, response.grounded));
      } catch (caught) {
        setStatus("error");
        setError(asAssistantError(caught));
        setHistory(pushHistory(nextQuery, null));
      }
    },
    [query],
  );

  const askAnother = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
    setLastQuestion(null);
    setQuery("");
  }, []);

  const replay = useCallback(
    (item: HistoryItem) => {
      setQuery(item.query);
      void submit(item.query);
    },
    [submit],
  );

  return useMemo(
    () => ({
      query,
      setQuery,
      status,
      result,
      error,
      lastQuestion,
      history,
      canSubmit,
      submit,
      askAnother,
      replay,
    }),
    [askAnother, canSubmit, error, history, lastQuestion, query, replay, result, status, submit],
  );
}
