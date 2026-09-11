import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { groundedFixture, insufficientFixture } from "./test/fixtures";

function jsonResponse(status: number, body: unknown): Promise<Response> {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response);
}

function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      return handler(url, init);
    }),
  );
}

describe("BIS Intelligent Assistant UI", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(500, { detail: "unhandled" });
    });
  });

  it("renders the main interface", async () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "BIS Intelligent Assistant" })).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(
      screen.getByText(/Grounded in the ingested pilot corpus/i),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByTestId("connection-status")).toHaveTextContent("API connected");
    });
  });

  it("keeps Ask disabled for a blank query", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Question"), "   ");
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("submits a question and shows a grounded answer with citations", async () => {
    const user = userEvent.setup();
    let finishAnswer: ((value: Response) => void) | undefined;
    const pendingAnswer = new Promise<Response>((resolve) => {
      finishAnswer = resolve;
    });
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return pendingAnswer;
    });

    render(<App />);
    await user.type(
      screen.getByLabelText("Question"),
      "What certification requirements exist for appliances?",
    );
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByTestId("loading-state")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retrieving…" })).toBeDisabled();
    finishAnswer?.(await jsonResponse(200, groundedFixture));

    expect(await screen.findByTestId("answer-panel")).toHaveTextContent("Grounded in retrieved clauses");
    expect(screen.getByTestId("citation-card")).toHaveTextContent("Sample QCO Circular");
    expect(screen.getByTestId("citation-card")).toHaveTextContent("Clause ID");
    expect(screen.getByRole("link", { name: "[1]" })).toHaveAttribute("href", "#citation-1");
    expect(screen.getByText(/method: vector/)).toBeInTheDocument();

    const fetchMock = vi.mocked(fetch);
    const answerCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/search/answer"));
    expect(answerCall?.[1]).toEqual(
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          query: "What certification requirements exist for appliances?",
        }),
      }),
    );
  });

  it("renders an insufficient-context response without inventing citations", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(200, insufficientFixture);
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "unrelated banana spaceship topic");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByTestId("insufficient-banner")).toBeInTheDocument();
    expect(screen.getByTestId("no-citations")).toHaveTextContent("No citations");
    expect(screen.queryByTestId("citation-card")).not.toBeInTheDocument();
    expect(screen.getByText(/does not establish an answer/i)).toBeInTheDocument();
  });

  it("shows API 500 errors", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(500, { detail: "Internal server error" });
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "What is covered?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    const alert = await screen.findByTestId("error-state");
    expect(within(alert).getByText(/Internal server error/)).toBeInTheDocument();
    expect(alert).toHaveTextContent("HTTP 500");
  });

  it("shows API 422 errors", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(422, {
        detail: [{ loc: ["body", "query"], msg: "query must not be blank", type: "value_error" }],
      });
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "valid looking question");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByTestId("error-state")).toHaveTextContent("query must not be blank");
  });

  it("shows API 400 errors", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(400, { detail: "Unknown retrieval method 'semantic'." });
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "What is covered?");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByTestId("error-state")).toHaveTextContent("Unknown retrieval method");
  });

  it("shows a network failure", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      throw new TypeError("Failed to fetch");
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "What is covered?");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByTestId("error-state")).toHaveTextContent("Could not reach the BIS assistant API");
  });

  it("Ask another question returns to the empty state", async () => {
    const user = userEvent.setup();
    mockFetch(async (url) => {
      if (url.endsWith("/health")) {
        return jsonResponse(200, { status: "ok", database: "ok" });
      }
      return jsonResponse(200, groundedFixture);
    });

    render(<App />);
    await user.type(screen.getByLabelText("Question"), "certification requirements");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByTestId("answer-panel");
    await user.click(screen.getByRole("button", { name: "Ask another question" }));
    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toHaveValue("");
  });
});
