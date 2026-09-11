import { describe, expect, it } from "vitest";

import { groundedFixture } from "../test/fixtures";
import { citationHeading, sourceForCitation } from "./citations";

describe("sourceForCitation", () => {
  it("joins by clause_id from the API response", () => {
    const source = sourceForCitation(groundedFixture.citations[0], groundedFixture.sources);
    expect(source?.clause_text).toContain("BIS certification");
  });

  it("does not invent a source when none match", () => {
    expect(
      sourceForCitation(
        { ...groundedFixture.citations[0], index: 9, clause_id: 999 },
        groundedFixture.sources,
      ),
    ).toBeUndefined();
  });
});

describe("citationHeading", () => {
  it("prefers document title from the citation object", () => {
    expect(citationHeading(groundedFixture.citations[0])).toBe(
      "Sample QCO Circular for Household Appliances",
    );
  });
});
