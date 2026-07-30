# Notes from the FINAL verify (R5+R6 integration, 2026-07-30, PASS_WITH_NOTES)

Fixed at close-out by the architect:

- [x] graphrag with Neo4j down was a bare 500 after ~30s of driver retries.
      -> MODES_NEEDING_A_GRAPH probe in generate.py refuses instantly; the API
         maps it to 503 with the `docker start finrag-neo4j` fix in the body.
         Tests in test_generate.py and test_api.py.
- [x] graph/retriever.py docstring still said "not registered". -> Corrected.
- [x] tests/graph/test_traverse.py ROLES was missing engineering, so the two
      role-parameterized leakage tests never ran for it. -> Added. (The verifier
      had already closed the hole manually on the real graph: 0 leak paths
      against a 35,517-path unmasked control.)
- [x] DECISIONS: nine files over the 250-line ceiling, not six; five ui/ files,
      not four. -> Both corrected.

Accepted, not fixed (know these exist):

- test_api.py test_hr_row_pii_never_appears_in_a_response holds on the micro
  fixture but over-promises for the real corpus: ingest writes employee names
  into HR row chunk TEXT, so hr and c_level (who may see those rows) do get
  names in responses. That is permitted access, not leakage; the test name is
  just broader than what it pins.
- At-rest exposure: the LLM graph build turns HR employee names into entity
  node labels in the shared :Finrag namespace. The app is safe (traversal
  returns chunk ids only; every edge touching such an entity carries an hr row
  source_chunk_id, so no non-hr role can walk to it), but anyone with direct
  Neo4j access sees the labels. The container is local dev only.
- Stale-vector window (same ids, changed text) remains open; recorded in
  DECISIONS under R6.
- ENV QUIRK, worse than recorded: `uv run` re-hides the .pth files on EVERY
  invocation, not only after sync. Console scripts (finrag) fail unless
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` runs in the same
  shell line; pytest is unaffected, which is why it stayed invisible. Runbook:
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth && .venv/bin/finrag ...`

# Punch list from the R2-R4 independent verify (2026-07-29, PASS_WITH_NOTES)

For the integrator. None of these touch the ABAC invariant; leakage is 0 everywhere.

Folded in at integration (R5 + R6), 2026-07-29. Notes are on each line.

- [x] dense.py: DenseRetriever loads dense.npz but throws the stored model name
      away. A Vertex-built matrix queried with FakeEmbedder returns plausible
      garbage (cosines ~0.05) instead of an error. Add the model comparison in
      __init__, mirroring embed_cache.py:51. HIGHEST VALUE FIX.
      -> `_check_model_matches_embedder` raises on a space mismatch (a cache can
         be refilled and starts empty, a matrix cannot, so this refuses).
- [x] sweep.py: "offline" is cache luck, not structure. build_embedder defaults
      to vertex; a cold cache or one new gold question makes the free gate bill
      API calls. Fail loudly on a cache miss (or pass a fake explicitly).
      -> `CacheOnlyEmbedder` + `build_sweep_embedder`. A miss names the question
         and points at `finrag eval --mode dense`. The fake path is untouched.
- [x] llm/vertex.py (R3's) has no tests at all. R6's branch ships its own
      vertex.py WITH 17 tests and end-to-end verification; at integration keep
      the tested one and adapt rerank.py to it.
      -> R6's file survives. The reranker's dials (thinking off, 512 tokens, one
         try) moved into `build_rerank_llm` there; rerank.py itself needed no
         change, since it only ever wanted an `LLMBackend`. 8 more tests.
- [x] dense.py:191: also check vectors.shape[0] == len(chunk_ids) so a
      hand-forged npz fails clean instead of IndexError at query time.
      -> In `load_matrix`, so every caller gets it, not only the retriever.
- [x] Stale-vector window: pipeline writes chunks.jsonl + bm25 before dense.npz;
      id-only guard accepts same-ids-changed-text. Note it in DECISIONS; a text
      hash in the npz is the real fix if it ever bites.
      -> Noted in DECISIONS under R6, still not fixed. It needs a change to what
         ingest writes, and no rung has been bitten by it yet.
- [x] rerank.py is 264 lines (over the 250 ceiling), undisclosed.
      -> Disclosed in the module docstring and in DECISIONS, with the split that
         was considered and why it would hurt. bm25.py (274) is disclosed too.
- [x] DECISIONS wording: cli.py is 343 lines not 333; "19 of 500 slots" should
      read "19 pairs (38 slots)".
      -> Both corrected in the R3 and R4 sections.

# Punch list from the R1 independent verify (2026-07-29)

Verdict: pass. R1 commit a62b94a reproduced clean in isolation (273/0, every eval
cell matched). These are the minor findings to fold in at integration time, not
before, because the R2-R4 workers are editing the tree right now.

## Code

- [x] `bm25.py _weight_mask`: add a one-line comment that the index method must
      stay `lucene`. With `bm25l`/`bm25+`, bm25s adds a nonoccurrence floor AFTER
      the mask multiply, so every denied chunk would get a nonzero score and the
      first lock would stop working. (Verifier proved each lock holds alone;
      control with both removed leaked 632 hits.)
      -> Added, naming the `score > 0.0` test as what would let them through.
- [x] `bm25.py load_index`: a present `params.index.json` with a missing
      `chunk_ids.json` (the half-written-artifact shape) falls through to a bare
      FileNotFoundError. Check both files in the same guard.
      -> One guard, both files, and the message names the missing one.
- [x] `bm25.py __init__`: the empty-chunks early-out skips
      `_check_index_matches_chunks`, so `BM25Retriever([], index_dir=stale)`
      accepts a mismatch silently. Unreachable today; reorder when the
      constructor is next touched.
      -> Reordered to match dense.py, with a test on the empty-list case.
- [x] `bm25.py` on disk has uncommitted, behavior-identical reflow on top of
      a62b94a. Fold into a later commit or drop it; do not leave it diverged.
      -> Committed with R2-R4 at 58ba4e7 before this integration started; the
         working tree and HEAD agreed at the merge.
- [x] `bm25.py` is exactly 250 lines (the ceiling). If it must grow, the build
      helpers (~lines 42-124) are the seam to split out.
      -> It grew to 274, all of it the two invariant comments asked for above.
         Disclosed rather than split: moving the build helpers away would put the
         lucene note in one file and the mask it protects in another.

## DECISIONS.md wording

- [x] T5 explanation: `1,654,767` is never one token (tokenizer splits it to
      1/654/767). The effect is real, the stated mechanism is wrong. Fix the
      sentence.
      -> Rewritten, with the correction kept visible rather than overwritten.
- [x] Ambiguous line: write "0.500 (2 of 4) -> 0.750 (3 of 4)" so the r0 column
      is not misread.
      -> Done.
- [x] Record the deferred-stopwords cost: function words alone hold top-5 slots
      today (rank 1 stays correct and well separated). R3's reranker inherits a
      noisy top-30 and should know why.
      -> Recorded in R1, under the dropped-alternatives list where the deferral
         was made.

## Forward decisions

- [x] R6: `Hit.debug` carries the exact bm25 float. Global IDF means scores leak
      information about invisible documents. Decide whether debug is role-gated
      before the API exposes it.
      -> Decided, and only half solved. `prompting.hit_view` builds each response
         hit from a fixed field list with no `debug` and no `payload`, so the
         exact bm25 float and the matched-term list never leave the process. But
         `score` is still sent, rounded to 4 places, and a bm25 score carries the
         same global IDF. The residue is small (one number per visible hit, no
         ids, no counts) and it is what makes a returned ranking checkable by the
         person reading it. Recorded in DECISIONS, R6, as accepted not fixed.
- [x] Process: commit rungs only AFTER their verifier passes (R1 was committed
      mid-verify; it happened to be clean). The R2-R4 flow already orders it
      correctly.
      -> Held for R5 and R6: both were built on branches, and the only commit
         this integration made is the r5-graph merge. Everything else is left
         dirty for the architect to review before it is committed.
