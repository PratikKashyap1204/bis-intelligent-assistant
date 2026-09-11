import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiUrl, askQuestion, errorFromResponse, fetchHealth } from "./client";

describe("apiUrl", () => {
  it("uses a relative path when no VITE_API_BASE_URL is set (Vite proxy)", () => {
    expect(apiUrl("/api/search/answer")).toBe("/api/search/answer");
    expect(apiUrl("/health")).toBe("/health");
  });
});

describe("errorFromResponse", () => {
  it("maps 422 validation arrays from FastAPI", () => {
    const error = errorFromResponse(422, {
      detail: [{ loc: ["body", "query"], msg: "query must not be blank", type: "value_error" }],
    });
    expect(error.kind).toBe("validation");
    expect(error.message).toContain("query must not be blank");
  });

  it("maps 400 string detail", () => {
    const error = errorFromResponse(400, { detail: "Unknown retrieval method 'semantic'." });
    expect(error.kind).toBe("bad_request");
    expect(error.message).toContain("Unknown retrieval method");
  });

  it("maps 500 generic detail", () => {
    const error = errorFromResponse(500, { detail: "Internal server error" });
    expect(error.kind).toBe("server");
    expect(error.message).toBe("Internal server error");
  });
});

describe("askQuestion", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("POSTs { query } to /api/search/answer", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        answer: "ok",
        citations: [],
        retrieval_method: "vector",
        sources: [],
        grounded: false,
        context_used: 0,
        evidence_status: "insufficient",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await askQuestion("What is covered?");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/search/answer",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ query: "What is covered?" }),
      }),
    );
  });

  it("throws a network error when fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(askQuestion("q")).rejects.toMatchObject({ kind: "network", status: null });
  });
});

describe("fetchHealth", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns parsed health JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok", database: "ok" }),
      }),
    );
    await expect(fetchHealth()).resolves.toEqual({ status: "ok", database: "ok" });
  });
});
