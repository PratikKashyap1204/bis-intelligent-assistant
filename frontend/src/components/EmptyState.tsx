export function EmptyState() {
  return (
    <section className="empty" data-testid="empty-state" aria-label="How this assistant works">
      <h2>Ask, then inspect the source</h2>
      <p>
        This is not a general chatbot. Each answer is produced only from clauses retrieved
        from the currently ingested BIS corpus, and every citation is copied from that
        retrieval — never invented in the browser.
      </p>
      <ul>
        <li>Pilot corpus: 5 Indian Standard metadata records, 3 QCO circulars, 144 clauses.</li>
        <li>Ingested clause text comes from those QCO/circular PDFs, not the paid full text of Indian Standards.</li>
        <li>If retrieval does not find enough evidence, the assistant will say so instead of guessing.</li>
      </ul>
    </section>
  );
}
