import type { AnswerResponse } from "../api/types";

/**
 * Synthetic fixtures matching backend test conventions (IS 9999 /
 * example.invalid). These are NOT live BIS answers.
 */
export const groundedFixture: AnswerResponse = {
  answer:
    "Based only on the retrieved BIS material below, here is what is explicitly supported by the retrieved clauses:\n\n" +
    '[1] QCO circular \'Sample QCO Circular for Household Appliances\' (relating to IS 9999), Clause 3 (CLAUSE), page 2: "All manufacturers of electrical appliances covered by this QCO are required to obtain BIS certification before such appliances may be sold in the domestic market."\n\n' +
    "Anything not explicitly stated in the quoted clauses above cannot be determined from the currently retrieved BIS material.",
  citations: [
    {
      index: 1,
      standard_number: "IS 9999",
      standard_title: "Household Electric Appliance Safety Requirements",
      document_title: "Sample QCO Circular for Household Appliances",
      document_type: "QCO",
      clause_number: "3",
      clause_type: "CLAUSE",
      page_number: 2,
      source_url: "https://example.invalid/sample-qco.pdf",
      relevance_score: 0.55,
      clause_id: 101,
      document_id: 7,
    },
  ],
  retrieval_method: "vector",
  sources: [
    {
      standard_number: "IS 9999",
      standard_title: "Household Electric Appliance Safety Requirements",
      document_title: "Sample QCO Circular for Household Appliances",
      document_type: "QCO",
      clause_number: "3",
      clause_type: "CLAUSE",
      clause_text:
        "All manufacturers of electrical appliances covered by this QCO are required to obtain BIS certification before such appliances may be sold in the domestic market.",
      page_number: 2,
      source_url: "https://example.invalid/sample-qco.pdf",
      relevance_score: 0.55,
      relevance_method: "vector",
      clause_id: 101,
      document_id: 7,
    },
  ],
  grounded: true,
  context_used: 1,
  evidence_status: "supported",
};

export const insufficientFixture: AnswerResponse = {
  answer:
    "The available BIS material does not establish an answer to this question. No sufficiently relevant clause was found in the currently ingested BIS Standard/QCO content for this query.",
  citations: [],
  retrieval_method: "vector",
  sources: [],
  grounded: false,
  context_used: 0,
  evidence_status: "insufficient",
};

export const partialFixture: AnswerResponse = {
  ...groundedFixture,
  answer:
    "Certification before sale is required [1]. Earthing details for this product are not in the supplied material.",
  grounded: true,
  evidence_status: "partially_supported",
};

export const outOfScopeFixture: AnswerResponse = {
  ...insufficientFixture,
  answer:
    "This question is outside the currently ingested BIS corpus. The available BIS material does not establish an answer to this question because the ingested Standard/QCO content does not cover this topic.",
  evidence_status: "out_of_scope",
};
