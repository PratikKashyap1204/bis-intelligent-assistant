import { useCallback, useEffect, useState } from "react";

import { fetchHealth } from "../api/client";

export type ConnectionState =
  | { kind: "checking" }
  | { kind: "ok"; database: string }
  | { kind: "degraded"; database: string; message: string }
  | { kind: "unreachable"; message: string };

export function useHealth(pollMs = 30_000): ConnectionState {
  const [state, setState] = useState<ConnectionState>({ kind: "checking" });

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const health = await fetchHealth(signal);
      if (health.status === "ok" && health.database === "ok") {
        setState({ kind: "ok", database: health.database });
        return;
      }
      setState({
        kind: "degraded",
        database: health.database,
        message: `Backend status: ${health.status}; database: ${health.database}.`,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      const message =
        error && typeof error === "object" && "message" in error
          ? String((error as { message: string }).message)
          : "Could not reach the backend health endpoint.";
      setState({ kind: "unreachable", message });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    const id = window.setInterval(() => {
      void refresh();
    }, pollMs);
    return () => {
      controller.abort();
      window.clearInterval(id);
    };
  }, [pollMs, refresh]);

  return state;
}
