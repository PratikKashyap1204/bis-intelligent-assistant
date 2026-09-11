import type { ConnectionState } from "../hooks/useHealth";

interface HeaderProps {
  connection: ConnectionState;
}

function statusLabel(connection: ConnectionState): { text: string; tone: string } {
  if (connection.kind === "checking") {
    return { text: "Checking backend…", tone: "is-muted" };
  }
  if (connection.kind === "ok") {
    return { text: "API connected", tone: "is-ok" };
  }
  if (connection.kind === "degraded") {
    return { text: "API reachable, database issue", tone: "is-warn" };
  }
  return { text: "API unreachable", tone: "is-bad" };
}

export function Header({ connection }: HeaderProps) {
  const status = statusLabel(connection);
  return (
    <header className="site-header">
      <div className="site-header__brand">
        <div className="mark" aria-hidden="true">
          BIS
        </div>
        <div>
          <p className="eyebrow">Bureau of Indian Standards · SIH pilot</p>
          <h1>BIS Intelligent Assistant</h1>
        </div>
      </div>
      <p
        className={`connection ${status.tone}`}
        data-testid="connection-status"
        title={
          connection.kind === "ok"
            ? "GET /health returned status=ok, database=ok"
            : connection.kind === "checking"
              ? "Requesting GET /health"
              : connection.message
        }
      >
        <span className="connection__dot" aria-hidden="true" />
        {status.text}
      </p>
    </header>
  );
}
