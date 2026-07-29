# Punch list from the R2-R4 independent verify (2026-07-29, PASS_WITH_NOTES)

For the integrator. None of these touch the ABAC invariant; leakage is 0 everywhere.

- [ ] dense.py: DenseRetriever loads dense.npz but throws the stored model name
      away. A Vertex-built matrix queried with FakeEmbedder returns plausible
      garbage (cosines ~0.05) instead of an error. Add the model comparison in
      __init__, mirroring embed_cache.py:51. HIGHEST VALUE FIX.
- [ ] sweep.py: "offline" is cache luck, not structure. build_embedder defaults
      to vertex; a cold cache or one new gold question makes the free gate bill
      API calls. Fail loudly on a cache miss (or pass a fake explicitly).
- [ ] llm/vertex.py (R3's) has no tests at all. R6's branch ships its own
      vertex.py WITH 17 tests and end-to-end verification; at integration keep
      the tested one and adapt rerank.py to it.
- [ ] dense.py:191: also check vectors.shape[0] == len(chunk_ids) so a
      hand-forged npz fails clean instead of IndexError at query time.
- [ ] Stale-vector window: pipeline writes chunks.jsonl + bm25 before dense.npz;
      id-only guard accepts same-ids-changed-text. Note it in DECISIONS; a text
      hash in the npz is the real fix if it ever bites.
- [ ] rerank.py is 264 lines (over the 250 ceiling), undisclosed.
- [ ] DECISIONS wording: cli.py is 343 lines not 333; "19 of 500 slots" should
      read "19 pairs (38 slots)".

# Punch list from the R1 independent verify (2026-07-29)

Verdict: pass. R1 commit a62b94a reproduced clean in isolation (273/0, every eval
cell matched). These are the minor findings to fold in at integration time, not
before, because the R2-R4 workers are editing the tree right now.

## Code

- [ ] `bm25.py _weight_mask`: add a one-line comment that the index method must
      stay `lucene`. With `bm25l`/`bm25+`, bm25s adds a nonoccurrence floor AFTER
      the mask multiply, so every denied chunk would get a nonzero score and the
      first lock would stop working. (Verifier proved each lock holds alone;
      control with both removed leaked 632 hits.)
- [ ] `bm25.py load_index`: a present `params.index.json` with a missing
      `chunk_ids.json` (the half-written-artifact shape) falls through to a bare
      FileNotFoundError. Check both files in the same guard.
- [ ] `bm25.py __init__`: the empty-chunks early-out skips
      `_check_index_matches_chunks`, so `BM25Retriever([], index_dir=stale)`
      accepts a mismatch silently. Unreachable today; reorder when the
      constructor is next touched.
- [ ] `bm25.py` on disk has uncommitted, behavior-identical reflow on top of
      a62b94a. Fold into a later commit or drop it; do not leave it diverged.
- [ ] `bm25.py` is exactly 250 lines (the ceiling). If it must grow, the build
      helpers (~lines 42-124) are the seam to split out.

## DECISIONS.md wording

- [ ] T5 explanation: `1,654,767` is never one token (tokenizer splits it to
      1/654/767). The effect is real, the stated mechanism is wrong. Fix the
      sentence.
- [ ] Ambiguous line: write "0.500 (2 of 4) -> 0.750 (3 of 4)" so the r0 column
      is not misread.
- [ ] Record the deferred-stopwords cost: function words alone hold top-5 slots
      today (rank 1 stays correct and well separated). R3's reranker inherits a
      noisy top-30 and should know why.

## Forward decisions

- [ ] R6: `Hit.debug` carries the exact bm25 float. Global IDF means scores leak
      information about invisible documents. Decide whether debug is role-gated
      before the API exposes it.
- [ ] Process: commit rungs only AFTER their verifier passes (R1 was committed
      mid-verify; it happened to be clean). The R2-R4 flow already orders it
      correctly.
