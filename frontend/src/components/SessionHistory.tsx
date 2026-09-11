import type { HistoryItem } from "../lib/history";

interface SessionHistoryProps {
  items: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  disabled: boolean;
}

export function SessionHistory({ items, onSelect, disabled }: SessionHistoryProps) {
  if (items.length === 0) {
    return null;
  }

  return (
    <section className="history" aria-label="Questions asked this session">
      <h2>This session</h2>
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              className="history__item"
              onClick={() => onSelect(item)}
              disabled={disabled}
            >
              <span className="history__query">{item.query}</span>
              <span className="history__meta">
                {item.grounded === true ? "Grounded" : item.grounded === false ? "Insufficient context" : "Error"}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
