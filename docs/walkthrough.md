# Walkthrough

The README says what the system does. This page shows it doing it: one policy
decision, one masked search, and one agent run traced end to end, including a
failure. Every artifact quoted here is in the repo.

## One policy decision

The rules live in `policy/policy.yaml`. Here is the pair that governs HR rows:

```yaml
- name: hr_rows_hr_and_c_level_only
  when:
    all:
      - eq: [resource.owner_dept, hr]
      - eq: [resource.doc_type, csv]
      - in: [action, [retrieve_chunk, view_source, view_row]]
      - in: [subject.role, [hr, c_level]]
  effect: permit

- name: non_hr_hr_rows_deny
  when:
    all:
      - eq: [resource.owner_dept, hr]
      - eq: [resource.doc_type, csv]
      - in: [action, [retrieve_chunk, view_source, view_row]]
      - not_in: [subject.role, [hr, c_level]]
  effect: deny
```

Ask the engine "may marketing retrieve chunk `hr/hr_data#row-FINEMP1000`" and
rule two fires: deny, by name. Ask the same for `hr` and rule one permits. No
rule matching at all is also a deny; the default is closed. The engine returns
the rule name with every decision, so a refusal in the API carries its reason
and the leakage sweep can report exactly which rule a planted leak breaks.

## One masked search

Every retriever follows the same shape. From `src/finrag/retrieval/bm25.py`:

```python
def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
    """Best `k` chunks this role may see, ranked by BM25 score."""
    ...
    # The mask comes FIRST, before any scoring or cutting.
    allowed = visible_ids(self.chunks, role)
    ...
    scores = self.index.get_scores(query_terms, weight_mask=self._weight_mask(allowed))

    scored = []
    for row, chunk in enumerate(self.chunks):
        # Second lock on the same door, deliberately. If the mask above were
        # ever passed wrongly, this loop still refuses denied chunks.
        if chunk.chunk_id not in allowed:
            continue
        ...
```

The order is the invariant: visibility first, scoring second, the cut to k
last. A denied chunk never occupies a rank slot, so masking cannot be undone
by trimming the list afterwards. Each retriever carries a second in-loop check,
and the eval harness re-judges every returned hit against the YAML rules,
trusting no stamp. The locks are tested by breaking each one and proving the
others still hold; with all of them removed, the control test leaks 632 hits.

The same rule holds where it is less obvious:

- the reranker masks before it builds its prompt, so denied text never reaches
  Gemini even as context to be discarded;
- the graph traversal filters every relationship inside the Cypher on the
  chunks the role may see, so a denied edge cannot even be walked through;
- the agent's observations pass the mask before they enter its prompt, so
  denied text never feeds its reasoning.

## One agent run, end to end

Mode `react` gives the model a loop instead of a pipeline: think, search, read,
search again, up to four hops, then write a cited answer. Both runs below are
in `evals/results/r7_answers_20260730T0255.json`.

### The win: T4-05

The question, asked as c_level:

> In Q3 2024 finance put 35 percent of $130 million of vendor services behind
> loyalty and retention. What did the loyalty programme return?

The answer needs two documents: the finance expense breakdown and the Q3
marketing report. `hybrid_rerank` returns NO_ANSWER here; one fixed retrieval
pass does not cover both halves well enough to satisfy its refusal rule.

The agent's hop queries, exactly as it wrote them:

1. `loyalty program return Q3 2024`
2. `loyalty program financial return Q3 2024`
3. `loyalty program revenue contribution Q3 2024`
4. `loyalty program financial impact Q3 2024`

And its answer:

> In Q3 2024, $45.5 million (35% of $130 million) was allocated to loyalty
> programs and customer retention strategies [1]. While loyalty programs
> contributed to an overall 30% year-over-year revenue increase [2], the
> specific financial return of the loyalty program itself is not detailed in
> the provided information. The loyalty program enrolled 40,000 customers,
> with 55% redeeming rewards by quarter-end [3].

Three citations, both documents, the computed figure, and an honest note about
what the corpus does not say. This one question is the argument for the mode.

### The failure: T4-03

> Finance flagged in Q2 2024 that influencer marketing might underperform
> against expected ROI. What ROI did marketing report?

The gold answer is 2.3x against a 2.4x target. `hybrid_rerank` finds it in one
pass. The agent's four hops:

1. `influencer marketing ROI Q2 2024`
2. `influencer marketing reported ROI Q2 2024`
3. `influencer marketing ROI target Q2 2024`
4. `influencer marketing specific ROI target Q2 2024`

Four rewordings of the same idea. The 2.3x figure sits in a benchmarks section
those words never reach, and the agent finished by saying the documents do not
specify the ROI. Five model calls to be more wrong than one.

This failure taught more than the win. When a hop finds nothing new, the agent
rephrases the whole question instead of decomposing it into a different fact.
The head-to-head on the ten multi-hop questions went 1 win, 3 losses, 4 ties,
2 both-fail against `hybrid_rerank`, which is why the recorded routing rule is:
`react` when the ranked list is the product, `hybrid_rerank` when the answer
is. The fix to try next is in the build log: show the agent what it has already
covered, not only what it has already asked.

## Reading the eval artifacts

Every run writes one JSON to `evals/results/`, named `{stage}_{mode}_{stamp}`.
Inside:

- `overall` and `per_tier`: recall, precision, MRR at the document level, and
  the headline `section_hit_rate` and `section_mrr`. Section scores answer
  "was the right part of the right document near rank 1".
- `per_question`: one record per gold question with the returned chunk ids, so
  any aggregate can be recomputed from the rows. Independent reviewers re-derive
  every table this way; a cell that cannot be reproduced fails the stage.
- `ambiguous`: the four contradiction probes, scored on finding both sources.
- `policy_probes`: the T6 questions whose right outcome is a refusal.
- `leakage_count`: computed by replaying every returned hit through the policy
  engine. The build gate requires exactly zero.

The sweep file (`r7_sweep_*.json`) is the wider net: every question, every
role, every mode, 2,100 cells, with a per-role, per-mode breakdown whose sums
match the totals. The suite plants a deliberately leaky retriever to prove the
sweep catches what it exists to catch.

## Known limits

Open items live in [`followups.md`](followups.md), each with what it would take
to close: the agent's answer text bypasses one of the outbound locks (masked
observations make it unreachable today), per-question hop counts are not saved
in the eval artifact, the embedding cache can be clobbered by a run with a
different embedder, and no mode yet surfaces both sides of a planted
contradiction. They are listed because an audit that ends "nothing left" was
not an audit.
