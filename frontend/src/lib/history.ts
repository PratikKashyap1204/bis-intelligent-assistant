const STORAGE_KEY = "bis-assistant-session-history";
const MAX_ITEMS = 12;

export interface HistoryItem {
  id: string;
  query: string;
  askedAt: string;
  grounded: boolean | null;
}

export function loadHistory(): HistoryItem[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isHistoryItem).slice(0, MAX_ITEMS);
  } catch {
    return [];
  }
}

function isHistoryItem(value: unknown): value is HistoryItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as HistoryItem;
  return typeof item.id === "string" && typeof item.query === "string" && typeof item.askedAt === "string";
}

export function pushHistory(query: string, grounded: boolean | null): HistoryItem[] {
  const nextItem: HistoryItem = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    query,
    askedAt: new Date().toISOString(),
    grounded,
  };
  const existing = loadHistory().filter((item) => item.query !== query);
  const next = [nextItem, ...existing].slice(0, MAX_ITEMS);
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Private mode / quota — history stays in-memory via React state.
  }
  return next;
}
