# Decisions

Build log: what each rung added, alternatives considered, tradeoffs, and metric deltas.

## R0: scaffold, contracts, chunker, policy engine, HR ingest, gold set, harness, naive baseline

### What was added

- Contracts everything else depends on: `Chunk` and `Hit` in `src/finrag/models.py`, the
  `Retriever` protocol in `retrieval/base.py`, the `LLMBackend` protocol in `llm/base.py`.
- Heading-aware markdown chunker with stable ids. `chunk_id = doc_id#section_slug[/pN]`
  derives from the path and heading tree only, never from offsets or size settings, so gold
  labels survive a re-chunk. A snapshot test pins the id list.
- Policy engine ported from the old `docs/policy.yaml`: `from_yaml`, `decide`,
  `allowed_roles`, `replay`. Deny by default, folder means owning department. A full
  role by resource matrix test covers all six roles.
- HR ingest: 100 CSV row chunks with PII kept out of the searchable text and held in
  `payload` for `view_row` holders only, plus 5 precomputed aggregate chunks.
- Ingest pipeline writing `artifacts/chunks.jsonl`, with `allowed_roles` filled in by
  running the policy engine rather than by a hand written table.
- 50 question tiered gold set with a validator, the eval metrics module, the harness, and a
  naive term overlap retriever that exists only to prove the plumbing end to end.

### Alternatives considered

- Chunking: the old repo's adaptive overlap machinery. Dropped. It made ids depend on size
  settings, which meant every retuning silently invalidated the gold labels.
- `allowed_roles`: a second lookup table mapping folders to roles. Dropped. Two sources of
  truth drift, and the point of the invariant is that one engine decides.
- HR PII: index the full row and filter at answer time. Dropped. If the text is in the
  index it can be retrieved, ranked, and echoed. Keeping it out of searchable text and in
  `payload` makes the leak structurally impossible rather than policed after the fact.
- Gold answers: reuse the old 22 question set. Dropped. Several of its answers contradict
  the corpus, so it could not be a baseline for anything.
- Baseline retriever: start straight at BM25. Dropped. A deliberately weak floor gives every
  later rung a number to beat, and it shakes out the harness before the ranking work starts.

### Tradeoff accepted

The naive retriever is not a serious ranker, so its scores flatter the doc level metric. It
was chosen for that reason: R0 buys a working pipeline and an honest measuring stick, not
retrieval quality. The cost is that some tiers already look strong, and their real
difficulty will only show once we compare modes at section level and on MRR.

### Baseline table (mode `naive`, rung r0, real run)

Doc level recall is primary, section level secondary. `n_scored` is the number of questions
in the tier that carry retrievable gold docs. T4 shows 6 of 10 because the 4 `ambiguous`
probes are scored separately on both source recall. T6 shows 2 of 8 because 6 of them are
policy probes scored on behavior, not on ranking.

| Tier | n | scored | recall@5 | recall@10 | MRR | section hit | leakage |
|---|---|---|---|---|---|---|---|
| T1 lexical | 8 | 8 | 1.000 | 1.000 | 0.938 | 1.000 | 0 |
| T2 paraphrase | 8 | 8 | 0.750 | 0.750 | 0.625 | 0.125 | 0 |
| T3 temporal | 10 | 10 | 0.900 | 1.000 | 0.783 | 0.800 | 0 |
| T4 multi-hop | 10 | 6 | 0.750 | 0.917 | 0.917 | 1.000 | 0 |
| T5 aggregation | 6 | 6 | 0.625 | 0.667 | 0.500 | 0.667 | 0 |
| T6 ABAC | 8 | 2 | 0.500 | 0.500 | 0.250 | 0.500 | 0 |
| OVERALL | 50 | 40 | 0.806 | 0.863 | 0.733 | 0.700 | 0 |

Ambiguous both source recall@10: 0.500 (2 of 4). Policy probes: 6 of 6 passed.
Total leakage: 0, target 0.

Corpus after ingest: 318 chunks over 11 documents. 213 markdown, 100 HR rows,
5 aggregates. By owner department: hr 105, engineering 81, general 56, marketing 50,
finance 26. Markdown chunk size in estimated tokens: min 26, median 106, mean 125,
max 400, none over the 480 hard cap.

### Surprises

- The naive floor is not a floor on T1 and T3. Term overlap reaches recall@10 of 1.000 on
  both. Doc level recall over an 11 document corpus is a soft target: any chunk from the
  right file counts, and each quarterly lives in its own file, so the temporal distractors
  barely bite at doc level. T3 section hit rate of 0.800 and MRR of 0.783 are the honest
  numbers there, and section level is where hybrid plus metadata will have to prove itself.
- T2 splits exactly as designed. Doc recall 0.750 but section hit rate 0.125, the lowest of
  any tier by a wide margin. Term overlap finds the right document and the wrong part of it.
  That gap is the clearest headroom in the run and it belongs to dense retrieval.
- T4 posts an MRR of 0.917, near T1's 0.938, which looks wrong for a multi-hop tier until
  you see that only 6 of its 10 questions are scored on ranking and those 6 name rare terms.
  The 4 ambiguous probes, scored properly on retrieving both conflicting sources, come in at
  0.500. That is the real multi-hop signal and it is weak.
- T5 is the weakest scored tier, MRR 0.500. The aggregate chunks are short and share few
  terms with questions phrased as comparisons, so overlap has little to grab.
- Leakage is 0 on every tier at R0, before the R4 sweep exists. The mask in `base.py` runs
  before ranking, so denied chunks never occupy a rank slot. R4 still has to prove this for
  every question by every role, not only the roles the gold set happens to ask with.
- Environment, not code: `uv run` writes the editable install `.pth` file with the macOS
  hidden flag set, and Python 3.13 skips hidden `.pth` files, so the `finrag` console script
  cannot import its own package. `pytest` is unaffected because `tests/conftest.py` already
  falls back to putting `src` on `sys.path`. CLI runs in this rung used
  `PYTHONPATH=src uv run finrag ...`. Worth a permanent fix before the CLI becomes part of
  anyone else's workflow.

## R0 fixes (same rung, post-verify)

Architect rulings applied after the first R0 baseline, before R1 locks the comparison
table. No new rung: same corpus, same gold set, same naive retriever, re-measured.

### Rulings applied

- Section level is now the headline metric. `metrics.section_rr` scores the first hit that
  lands inside a gold section as `1 / its position in the returned list`, and `summarise`
  averages it as `section_mrr`. Positions count hits, not documents, so every returned chunk
  takes up a slot. Reason: after the role mask most roles see between one and six documents,
  so doc level recall is nearly free. The naive floor already posts recall@10 of 1.000 on T1,
  T3 and T6. Only section level still separates one retriever from another.
- The eval table prints section hit rate, section MRR, doc recall@10, doc MRR, leakage, in
  that order. Doc numbers stay in the report; they no longer lead it.
- The section prefix match is tightened. A gold prefix `p` matches a chunk id `c` only when
  `c == p`, or `c` starts with `p + "/"`, or `c` starts with `p + "--"`. Plain `startswith`
  let `doc#q4-rev` claim `doc#q4-revenue`, a different section that merely shares a word
  start. The two accepted continuations are the part suffix a split section gets and the
  separator the chunker uses between a heading and its subheading, so part suffixes and
  heading subtrees still match. Checked against the shipped gold set: no prefix lost a match.
- Honest empties. `section_hit_rate` and `section_mrr` are null when nothing in the group
  labelled a section, and `policy_summary.pass_rate` is null when the group holds no probes.
  A group that was never measured is not a group that scored zero, and the printed table
  shows a dash rather than 0.000.
- `finrag eval` now validates the gold set. It calls `goldset.load_gold` and `validate`
  against the built chunks and the real policy engine, and exits 1 with the whole problem
  list if anything drifted. The local `read_gold`, which read the file and asked no
  questions, is gone. Reason: a renamed heading or a dropped document used to turn every
  score into nonsense without failing anything.
- `harness.py` was over the 250 line limit, so `report_filename` and `write_report` moved to
  the new `eval/report_io.py`. Behaviour unchanged, imports updated.
- New `tests/test_naive.py`: the mask is proved directly. A chunk carrying all seven query
  words but denied to the asking role must not appear even at `k=10`, wider than the whole
  visible pool of two, so no cut-off is doing the hiding. Also tie-break stability under a
  reversed corpus, empty query returns nothing, and the registry's unknown-mode message.
- New `tests/test_cli.py`: eval refuses a broken gold set, refuses an unknown mode, refuses
  to run before ingest, and one end to end test runs ingest then eval on the real corpus and
  the real gold set in a scratch directory, offline, naive mode only.

### Fresh baseline (mode `naive`, rung r0, real run)

Section columns first. `scored` is the number of questions in the tier that are scored on
ranking; T4 shows 6 of 10 because 4 are ambiguous probes scored on both-sources recall, and
T6 shows 2 of 8 because 6 are policy probes scored on behaviour.

| Tier | n | scored | section hit | section MRR | recall@5 | recall@10 | doc MRR | leakage |
|---|---|---|---|---|---|---|---|---|
| T1 lexical | 8 | 8 | 1.000 | 0.740 | 1.000 | 1.000 | 0.938 | 0 |
| T2 paraphrase | 8 | 8 | 0.125 | 0.016 | 0.750 | 0.750 | 0.625 | 0 |
| T3 temporal | 10 | 10 | 0.800 | 0.625 | 0.900 | 1.000 | 0.783 | 0 |
| T4 multi-hop | 10 | 6 | 1.000 | 0.778 | 0.750 | 0.917 | 0.917 | 0 |
| T5 aggregation | 6 | 6 | 0.667 | 0.352 | 0.625 | 0.667 | 0.500 | 0 |
| T6 ABAC | 8 | 2 | 1.000 | 0.225 | 1.000 | 1.000 | 0.500 | 0 |
| OVERALL | 50 | 40 | 0.725 | 0.488 | 0.831 | 0.887 | 0.746 | 0 |

Ambiguous both source recall@10: 0.500 (2 of 4). Policy probes: 6 of 6 passed.
Total leakage: 0, target 0. Corpus unchanged at 318 chunks over 11 documents.

### Delta against the first baseline

| Tier | section hit | recall@10 | doc MRR |
|---|---|---|---|
| T1 | 1.000, no change | 1.000, no change | 0.938, no change |
| T2 | 0.125, no change | 0.750, no change | 0.625, no change |
| T3 | 0.800, no change | 1.000, no change | 0.783, no change |
| T4 | 1.000, no change | 0.917, no change | 0.917, no change |
| T5 | 0.667, no change | 0.667, no change | 0.500, no change |
| T6 | 0.500 to 1.000 | 0.500 to 1.000 | 0.250 to 0.500 |
| OVERALL | 0.700 to 0.725 | 0.863 to 0.887 | 0.733 to 0.746 |

Every movement is T6, and all of it comes from the policy fix in the same round of
verification. The HR CSV rules used to permit `view_row` alone, so the HR role could open a
row it was never allowed to retrieve, and T6-02, the positive control that asks HR for
Aadhya Patel's salary, scored 0 on a question it is supposed to answer. The rules now cover
`retrieve_chunk` and `view_source` as well, and T6-02 returns the row: recall@10 1.000,
section RR 0.250. The overall gain of 0.025 on section hit is exactly one question in forty.

The tightened prefix match moved nothing on this corpus, which is the result we wanted: it
closes a hazard without quietly repricing the baseline that R1 will be compared against.

### Surprises

- Section MRR is the metric the plan needed and did not have. Section hit rate says T1 and
  T4 are solved at 1.000; section MRR says T1 is 0.740 and T4 is 0.778, so the right passage
  is often sitting at rank 2 or 3 rather than rank 1. T2 is the starkest: hit rate 0.125,
  section MRR 0.016, meaning the one paraphrase question that finds its section finds it
  near the bottom of ten. Doc recall@10 on the same tier is 0.750. That gap is the honest
  size of the headroom, and it is much larger than the first baseline suggested.
- T6 section MRR is 0.225 even though its hit rate is 1.000. Both positive controls find
  their section at rank 4 or 5, behind chunks from the employee handbook and the finance
  summary that share more surface words. Term overlap gets there but does not rank it.
- HR row searchable text lost joining dates in the same round. `date_of_joining` moved out
  of `SEARCHABLE_FIELDS` and into the payload, because a joining date pins a named person to
  a date and nobody searches by one. The tenure aggregates read it straight off the CSV, so
  no metric depended on it, and no gold question moved.
- Housekeeping caught during verification: three macOS conflict copies had appeared in the
  tree (`tests/test_policy 2.py`, `tests/test_hr_csv 2.py`, `src/finrag/eval/metrics 2.py`).
  They were stale snapshots taken before the policy and HR edits, and pytest was collecting
  the two test files, so the suite was running old assertions against new code. Deleted.
- The `.pth` problem from R0 is not a Python 3.13 problem and is not fixed by the 3.12
  rebuild. `uv run` marks the editable install's `.pth` file hidden on every invocation, and
  this CPython build skips hidden `.pth` files by design. So `uv run finrag ...` cannot
  import its own package on this machine. Working commands, now in the README: clear the
  flag once with `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`, then run with
  `UV_NO_SYNC=1` so the sync step does not put it back. `pytest` is unaffected. This needs a
  real fix in `pyproject.toml` or a Makefile before R1.

## R1: the bm25 retrieval mode

### What was added

- `src/finrag/retrieval/bm25.py`: `BM25Retriever(chunks, index_dir=None)`, mode name `bm25`,
  plus the index build and save helpers ingest calls. Naming an `index_dir` loads the saved
  artifact, which is the path the CLI takes; leaving it out builds the index in memory, which
  is how the tests compare a loaded index against a fresh one.
- Ingest now emits a second artifact. After writing `chunks.jsonl`, `run_ingest` builds the
  BM25 index over the same list in the same order and saves it to `artifacts/bm25/`, so index
  row N is always line N of the chunks file. The folder holds the five bm25s files plus our
  own `chunk_ids.json`.
- One tokenizer for both sides. `retrieval/bm25.py` imports `tokenize` from `naive.py` rather
  than writing its own, and a test asserts they are the same function object. bm25s looks each
  query word up in the vocabulary built at index time, so a word split or folded differently
  at query time is simply not found and its evidence disappears with no error at all. The
  index build lives in the retrieval module for the same reason: the two uses cannot drift
  apart if there is only one function.
- The role mask still comes first, and now twice over. `search` computes `visible_ids` before
  it scores anything, hands bm25s a `weight_mask` that zeroes every denied row, and then also
  skips denied ids in the loop that collects candidates. Two locks on one door: a mask passed
  wrongly is a data leak, so the second check earns its keep.
- `registry.py` registers `bm25` and `available_modes()` now returns `["bm25", "naive"]`.
  `IndexBundle.bm25_dir` is wired through, `finrag ingest` prints the index line.
- `tests/test_bm25.py`, 26 tests: mask before rank proved directly, IDF sanity stated as the
  comparison with naive, determinism, k-cut, empty and unknown-word queries, empty corpus,
  the save and load round trip, byte-identical rebuilds across two processes, the row-order
  guard, and the registry.

### Alternatives considered

- Hand-rolled BM25, which is what the old repo did. Dropped. It rebuilt the whole index on
  every query, which is why its latency scaled with corpus size for no reason, and a
  hand-written scorer is a second thing to test that adds nothing. `bm25s` persists to disk,
  is plain numpy with no torch and no downloads, and its scoring is the standard lucene
  variant rather than one we have to defend.
- `bm25s.tokenize`, the library's own tokenizer with stopwords and stemming. Dropped for now.
  It would help paraphrase questions a little and would make this rung's numbers
  incomparable with R0's, because the naive floor tokenizes plainly. Sharing one tokenizer
  keeps the R0 to R1 delta a measurement of ranking alone. Stemming can arrive later as its
  own change with its own row in this table.
- `index.retrieve()` instead of `get_scores()`. Dropped. `retrieve` does its own top-k, so
  the mask and the tie-break would sit inside library code. `get_scores` returns the score
  vector and we cut it ourselves in plain Python that can be read and tested.
- One index per role, six in all, so BM25 statistics could never see a document the role
  cannot. Dropped as too expensive for the size of the hazard. See the tradeoff below.
- Letting bm25s number the vocabulary. Dropped, because it numbers tokens by walking a
  Python `set` and set order for strings changes with the interpreter's hash seed. The scores
  are identical either way, but the saved files came out byte-different on every rebuild.
  We now build the vocabulary ourselves in alphabetical order and hand bm25s the numbers.

### Tradeoff accepted

IDF and average document length are global, computed over all 318 chunks, not over the
chunks the asking role may see. So a denied chunk cannot occupy a rank slot, cannot influence
which visible chunk wins, and cannot be part of any answer, but it does contribute to how
rare a word is considered. In principle that is a channel: with enough queries and exact
scores, a role could infer something about how often a word appears in documents it cannot
read. In practice the API returns ranks and text, the corpus is a fixed ten documents, and
the alternative is six indexes to build, save, load and keep in step, along with six
vocabularies whose sizes themselves leak. R4's leakage sweep asserts what the invariant
actually promises, that no denied chunk is ever returned to any role for any question, and
this note records what it does not promise.

### Per-tier table, r1 bm25 next to r0 naive, both from real runs

Section columns are the headline. `scored` is the number of questions in the tier scored on
ranking: T4 shows 6 of 10 because 4 are ambiguous probes scored on both-sources recall, T6
shows 2 of 8 because 6 are policy probes scored on behaviour.

| Tier | n | scored | section hit | section MRR | recall@10 | doc MRR | leakage |
|---|---|---|---|---|---|---|---|
| T1 lexical | 8 | 8 | 1.000 to 1.000 | 0.740 to 0.792 | 1.000 to 1.000 | 0.938 to 0.854 | 0 to 0 |
| T2 paraphrase | 8 | 8 | 0.125 to 0.000 | 0.016 to 0.000 | 0.750 to 1.000 | 0.625 to 0.588 | 0 to 0 |
| T3 temporal | 10 | 10 | 0.800 to 0.800 | 0.625 to 0.700 | 1.000 to 1.000 | 0.783 to 0.950 | 0 to 0 |
| T4 multi-hop | 10 | 6 | 1.000 to 1.000 | 0.778 to 0.917 | 0.917 to 0.917 | 0.917 to 1.000 | 0 to 0 |
| T5 aggregation | 6 | 6 | 0.667 to 1.000 | 0.352 to 0.774 | 0.667 to 1.000 | 0.500 to 1.000 | 0 to 0 |
| T6 ABAC | 8 | 2 | 1.000 to 1.000 | 0.225 to 1.000 | 1.000 to 1.000 | 0.500 to 1.000 | 0 to 0 |
| OVERALL | 50 | 40 | 0.725 to 0.750 | 0.488 to 0.637 | 0.887 to 0.988 | 0.746 to 0.876 | 0 to 0 |

Ambiguous both-sources recall@10: 0.500 to 0.750, 3 of 4. Policy probes: 6 of 6 passed in
both runs. Total leakage 0 in both, target 0. Corpus unchanged at 318 chunks over 11
documents; the index covers all 318.

Overall section MRR rises from 0.488 to 0.637, a gain of 0.149, which is the number that
matters: the right passage moved up the list. Doc recall@10 rises from 0.887 to 0.988 but
that metric is nearly saturated and should not be read as progress.

### Where this lands against what the plan expected

The plan expected T1 and T3 strong, T2 weak, T4 near zero, leakage 0. Three of those hold.
T4 does not, and it is worth saying plainly rather than filing it as a win.

- T1 strong, as expected, but not by the metric we predicted. Section MRR rises 0.740 to
  0.792 while doc MRR FALLS 0.938 to 0.854, driven by one question, T1-08. It names the
  annual marketing report and asks for CAC; BM25 answers with the near-identical quarterly
  reports, which repeat the same words in shorter documents, and length normalisation rewards
  them for it. Doc RR on that question drops from 1.000 to 0.333. This is the quarter
  collision the plan built T3 for, showing up in T1, and it is R3's job (hybrid plus
  metadata), not something BM25 can fix by itself.
- T3 improves where it should. Section MRR 0.625 to 0.700, doc MRR 0.783 to 0.950. T3-02,
  T3-06, T3-09 and T3-10 all moved up. T3-07 regressed, 1.000 to 0.500 on both metrics, the
  same near-identical-quarterly failure as T1-08.
- T2 is weak, as expected, and slightly worse than the floor. Section hit rate goes 0.125 to
  0.000: not one paraphrase question puts a gold section anywhere in its top ten. The one
  question naive got, T2-05, is a maternity leave paraphrase where the handbook shares the
  word "wage"; BM25 spends that credit on common words instead. T2-01 is the clearest case,
  "buyer attrition during the festive stretch" against a corpus that says churn and Q4, where
  doc RR falls 1.000 to 0.250. Nothing lexical can close this. It is R2's job and the tier is
  now at a true zero it can be measured against.
- T4 is the contradiction. The plan said near zero; the run says section MRR 0.917 and doc
  MRR 1.000, the strongest scored tier. Two reasons, neither of them multi-hop reasoning.
  Only 6 of the 10 questions are scored on ranking, and those 6 name rare terms that BM25 is
  built to find. The 4 questions that actually test the cross-document bridge are the
  ambiguous probes, and they sit at 0.750, 3 of 4. Reading T4's section MRR as multi-hop
  success would be reading the wrong 6 questions. Graph RAG at R5 has to be judged on the
  ambiguous probes and on whether it can answer a bridge question whose two halves share no
  vocabulary, not on this row.
- T5 is the honest surprise, section MRR 0.352 to 0.774. The aggregate chunks are short and
  hold rare tokens (`1,654,767`, department names), which is precisely the shape BM25 is good
  at, and length normalisation stops the 100 HR row chunks from crowding them out. T5-02 and
  T5-03 went from 0.000 to 1.000.
- T6 section MRR 0.225 to 1.000. Both positive controls now come back at rank 1. Leakage
  stays 0 and all 6 policy probes pass, but R4 still has to prove that for every role, not
  only the roles the gold set happens to ask with.

### Surprises

- bm25s does not write a timestamp, but its saved index was still byte-different on every
  rebuild. The cause is vocabulary numbering: bm25s assigns token ids by iterating a Python
  `set`, and string hashing is randomised per process, so the CSC matrix came out in a
  different column order each run while scoring identically. We now number the vocabulary
  ourselves in alphabetical order and pass bm25s `(rows_of_token_numbers, vocab)`, which is a
  path it already supports. Two `finrag ingest` runs in separate processes now produce
  identical bytes for all six files, and `test_two_separate_processes_save_the_same_bytes`
  runs two fresh interpreters with different `PYTHONHASHSEED` values to prove it. An
  in-process test would have passed before the fix and told us nothing.
- `params.index.json` does record the bm25s version. That is fixed by the installed library
  rather than by the run, so it does not make two builds differ, and an upgrade is meant to
  change the artifact.
- bm25s cannot index an empty corpus: average document length is a division by zero. The
  retriever skips the index and returns nothing when there are no chunks, and `run_ingest`
  now fails with a plain message if the corpus produced no chunks at all, since that is a
  mis-set path and not a valid run.
- The saved index knows only row numbers, so a chunks file in a different order would attach
  one chunk's score to another chunk with nothing raised. `chunk_ids.json` records the row
  order and loading refuses a mismatch. That failure would not have crashed; it would have
  produced plausible wrong numbers, which is worse.
- One test outside this rung's files had to change. `tests/test_naive.py` used `"bm25"` as its
  example of an unknown mode, which stopped being unknown. It now asks for `"telepathy"`, a
  name no rung will add.
- The `.pth` hidden-flag problem is unchanged and still needs the `chflags` line before every
  CLI run. Both commands in this rung were run that way.

## R2: the dense retrieval mode, Vertex AI embeddings

### Decision

Add a third mode, `dense`: cosine similarity over Vertex AI `text-embedding-005`
vectors, 768 numbers per chunk, built once at ingest into `artifacts/dense.npz` and
read back at query time. The role mask runs first, exactly as in `bm25.py`, and the
query is the only thing embedded per search.

This rung exists for one number. R1 left T2, the paraphrase tier, at a section hit
rate of 0.000: not one of the eight questions put a gold section anywhere in its top
ten. Those questions were written to share no useful word with their answers, so no
amount of BM25 tuning could reach them. Either embeddings move T2 off zero or dense
retrieval is not worth its cost.

### What was added

- `src/finrag/retrieval/embed.py`: the `Embedder` protocol (`embed_texts` returning
  float32 rows that are already L2-normalised), `VertexEmbedder` over the google-genai
  SDK, and `FakeEmbedder`, whose vectors come from a sha256 of the text. Batching caps
  a call at 100 texts and 60,000 characters, and a failed call is retried three times
  two seconds apart before it gives up loudly.
- `src/finrag/retrieval/embed_cache.py`: `EmbeddingCache`, text to vector on disk keyed
  by sha256 of the exact text, and `CachingEmbedder`, which only asks the backend for
  texts the cache lacks. Both ingest and query time go through one of these over
  `artifacts/embed_cache.npz`.
- `src/finrag/retrieval/dense.py`: `DenseRetriever`, plus the `build_and_save_matrix`
  and `load_matrix` helpers ingest calls. Same shape as `bm25.py` on purpose: name a
  `dense_path` and the saved matrix is loaded, leave it out and the chunks are embedded
  in memory, which is how the tests compare the two.
- Ingest writes a third artifact. `run_ingest` builds the dense matrix over the same
  chunk list in the same order as `chunks.jsonl` and the BM25 index, and `dense.npz`
  carries the chunk ids in row order, so loading refuses a mismatch the way the BM25
  index already does.
- `config.load_env`, twelve lines, reads `.env` into `os.environ` without overwriting
  anything already set. Called once in `cli.main`. No python-dotenv: our `.env` is
  plain `KEY=VALUE` lines and one dependency for six lines of parsing is not worth it.
- `registry.py` registers `dense` and `available_modes()` now returns
  `["bm25", "dense", "naive"]`. `IndexBundle` gains an `embedder` field, and the CLI
  fills it in only for modes that need it.
- `tests/test_embed.py` (27 tests), `tests/test_embed_cache.py` (14), `tests/test_dense.py`
  (27). All offline. Mask before cut proved directly, tie-break determinism, the cache
  proved by counting backend calls rather than comparing vectors, the row-order guard,
  the batching caps, and the retry.

### Alternatives considered

- The deprecated `vertexai` SDK, which is what the old repo used. Dropped; `google-genai`
  is the supported client and takes `vertexai=True` for the same endpoint.
- A local sentence-transformers model. Dropped for now. It would remove the credential
  and the API cost, but it drags in torch, downloads weights on first run, and its
  numbers would not be the numbers the served API produces. The `ml` dependency group is
  still there if a later rung wants an offline comparison.
- No cache, re-embed on every ingest. Dropped. Every re-chunk would pay again, and worse,
  a repeated eval run would not be reproducible offline, so nobody could re-score a
  finished rung without credentials.
- A cache keyed by chunk id. Dropped. A chunk id survives an edit to the chunk's text,
  so the cache would happily serve the vector of a paragraph that no longer exists.
  sha256 of the text means a changed chunk is a miss by construction.
- Storing the cache as JSON. Dropped for a small npz: 318 vectors of 768 floats is
  1.0 MB as npz against roughly 5 MB of JSON digits, and one read instead of a parse.
- Building the embedder inside the registry, so `dense` mode needed no wiring. Dropped.
  Constructing one may open a Vertex client, and a process that only asked for `naive`
  should not need a credential. The CLI decides, `make_dense` refuses a bundle without
  an embedder and says which function builds one.
- A `FakeEmbedder` that hashes word or character trigrams, like the existing
  `FakeEmbeddings` in `llm/fake.py`. Dropped. Trigram vectors put lexically similar
  texts near each other, which quietly turns the offline dense tests into a second BM25
  test. Hashing the whole text gives unrelated vectors unless the text is identical,
  and that is a sharper instrument: a chunk whose text IS the query scores a cosine of
  exactly 1.0, the highest there is, so making that chunk the denied one proves the mask
  runs before the ranking and not after it.
- Embedding all 50 gold queries in one batch at eval start. Dropped as premature. The
  first run makes 50 one-text calls and takes 8 seconds; every run after that makes none.

### Tradeoff accepted

Chunk vectors are computed over the whole corpus, with no role in sight, and one matrix
serves all six roles. This is the same tradeoff R1 recorded for global IDF, and it is
smaller here: a vector depends only on its own chunk's text, so a denied chunk cannot
influence a visible chunk's score at all, only take a row in a file nobody may read
from. The mask still decides which rows are ever scored. R4's leakage sweep asserts what
the invariant promises.

Second, and more real: the eval and the API must share one embedding space. The cache
records which model wrote it and comes back empty for another, and `dense.npz` records
the space too, so a model change cannot quietly mix two kinds of vector in one matrix.
What none of that catches is a model deprecated under us and silently swapped
server-side. Nothing here would notice; the numbers would just move.

### Per-tier table, r2 dense next to r1 bm25 and r0 naive, all three from real runs

Section columns are the headline. `scored` is the number of questions in the tier scored
on ranking: T4 shows 6 of 10 because 4 are ambiguous probes scored on both-sources
recall, T6 shows 2 of 8 because 6 are policy probes scored on behaviour.

| Tier | n | scored | section hit (naive / bm25 / dense) | section MRR (naive / bm25 / dense) | leakage |
|---|---|---|---|---|---|
| T1 lexical | 8 | 8 | 1.000 / 1.000 / 0.875 | 0.740 / 0.792 / 0.812 | 0 / 0 / 0 |
| T2 paraphrase | 8 | 8 | 0.125 / 0.000 / 0.875 | 0.016 / 0.000 / 0.729 | 0 / 0 / 0 |
| T3 temporal | 10 | 10 | 0.800 / 0.800 / 0.900 | 0.625 / 0.700 / 0.653 | 0 / 0 / 0 |
| T4 multi-hop | 10 | 6 | 1.000 / 1.000 / 0.833 | 0.778 / 0.917 / 0.833 | 0 / 0 / 0 |
| T5 aggregation | 6 | 6 | 0.667 / 1.000 / 1.000 | 0.352 / 0.774 / 0.889 | 0 / 0 / 0 |
| T6 ABAC | 8 | 2 | 1.000 / 1.000 / 1.000 | 0.225 / 1.000 / 1.000 | 0 / 0 / 0 |
| OVERALL | 50 | 40 | 0.725 / 0.750 / 0.900 | 0.488 / 0.637 / 0.780 | 0 / 0 / 0 |

Doc level, same three runs, kept for the record and not read as progress:

| Tier | recall@10 (naive / bm25 / dense) | doc MRR (naive / bm25 / dense) |
|---|---|---|
| T1 | 1.000 / 1.000 / 0.875 | 0.938 / 0.854 / 0.875 |
| T2 | 0.750 / 1.000 / 1.000 | 0.625 / 0.588 / 0.938 |
| T3 | 1.000 / 1.000 / 1.000 | 0.783 / 0.950 / 0.850 |
| T4 | 0.917 / 0.917 / 0.833 | 0.917 / 1.000 / 0.917 |
| T5 | 0.667 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| T6 | 1.000 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| OVERALL | 0.887 / 0.988 / 0.950 | 0.746 / 0.876 / 0.912 |

Ambiguous both-sources recall@10: 0.500 naive, 0.750 bm25, 0.500 dense. Policy probes:
6 of 6 passed in all three runs. Total leakage 0 in all three, target 0. Corpus unchanged
at 318 chunks over 11 documents; the matrix covers all 318.

Overall section MRR rises 0.637 to 0.780, a gain of 0.143, almost the same size as R1's
gain of 0.149 over the naive floor. Section hit rate rises 0.750 to 0.900.

### What moved, and why

- T2 is the rung. Section hit rate 0.000 to 0.875, section MRR 0.000 to 0.729. Seven of
  the eight paraphrase questions now find a gold section, and five of them at rank 1.
  T2-02 ("if our main server hall burned to the ground, how long before customers could
  pay again") lands the RTO/RPO section at rank 1 against a corpus that never says "burned"
  or "server hall". T2-03, T2-06, T2-07 and T2-08 do the same. This is the single largest
  movement of any kind in the build so far, and it is the movement the tier was written
  to detect. The one still missing is T2-04, "which three nations were named as fresh
  territory in the April to June plan". It is not a paraphrase failure: doc recall@10 is
  1.000 and six of its ten hits are Q2 sections. The gold `q2-marketing-overview` section
  is simply not one of them, and rank 1 is the Q3 projections section, 0.0024 ahead of the
  Q2 one. Right document, wrong section, and a cross-quarter leader decided by the third
  decimal place.
- T5 keeps R1's gain and improves the ranking, section MRR 0.774 to 0.889. T5-05, a
  four-quarter conversion-rate comparison, went 0.143 to 1.000: the aggregate-style
  question matches an aggregate-style chunk on meaning where BM25 had to hope a number
  string collided.
- T4's scored half improved on the questions that need a bridge and lost the one that
  needs a document name. T4-03, T4-07, T4-08 and T4-09 all rose, three of them to 1.000.
  T4-01 fell 1.000 to 0.000 and is the honest bad case; see the surprises.
- T6 holds at section MRR 1.000 with leakage 0 and 6 of 6 policy probes, which is what a
  new mode is supposed to do to the ABAC tier: nothing.
- T1 and T3 dipped where R1's plan said they might, and by less than expected. T1 section
  hit 1.000 to 0.875, one question; T3 section MRR 0.700 to 0.653, and T3 section hit
  actually rose, 0.800 to 0.900. Two T1 questions that BM25 put at rank 2 (T1-03, T1-04)
  now come back at rank 1.

### Surprises

- The quarter collision came back wearing different clothes, and it is now the largest
  remaining problem. R1 recorded that BM25 confuses the four near-identical quarterly
  marketing reports because they repeat each other's words. Dense confuses them because
  they repeat each other's meaning, which is worse, because there is no rare token left
  to break the tie. T1-08 asks for CAC "according to the 2024 annual marketing report"
  and the top ten are ten quarterly sections scored between 0.7259 and 0.7610, a spread
  of 0.035, with the annual report nowhere in the list. T4-01 is the same shape: it needs
  a finance risks section and a Q4 campaign section, and gets `campaign-highlights--
  digital-marketing` from all four quarters plus three executive summaries. Both scored
  1.000 or better under BM25.
- The flat score band is the diagnostic, and it is measurable. Across all 50 questions
  the top-ten cosine range is on average 13.9 percent of the top score; BM25's is 49.6
  percent. Mean rank-1 cosine is 0.689, and near-misses sit two or three thousandths
  behind it. So dense is confident about the neighbourhood and nearly indifferent inside
  it. That is a good argument for R3 being fusion plus a metadata filter rather than
  either signal tuned harder: BM25 has the sharp opinions dense lacks, and a quarter
  label is a field on the chunk that neither retriever is allowed to see.
- T2's win is bigger than doc recall could ever have shown. Doc recall@10 on T2 was
  already 1.000 under BM25, unchanged at 1.000 under dense. Read only the doc columns and
  this rung did nothing on its own tier. The R0 ruling that made section level the
  headline is what makes the entire rung visible.
- The ambiguous probes went backwards, 3 of 4 to 2 of 4, on T4-09 ("what does it cost to
  acquire one new customer", whose two sources are the annual report and the Q1 report).
  Its section RR rose 0.500 to 1.000 at the same time. Dense found one source and ranked
  it first; it did not find the second. A metric that rewards ranking and a metric that
  rewards coverage came apart on one question, which is a reminder that the ambiguous
  probes, not T4's section MRR, are the multi-hop signal R5 will be judged on.
- `FakeEmbedder` had to be less clever than the fake it sits beside. The obvious build is
  trigram hashing, like `llm/fake.py`, so that similar texts get similar vectors. That
  would have made every offline dense test a lexical test in disguise, and it would have
  made the mask test weaker: with graded scores, showing that the denied chunk was really
  the top scorer takes an argument. Hashing the whole text makes it arithmetic instead.
  The bait's text is the query verbatim, cosine 1.0, and nothing else can reach it.
- The cache pays for itself immediately, not eventually. First ingest, 318 texts in 4
  batched calls, 5.7 seconds. Second ingest, byte-identical matrix, 0.13 seconds and zero
  calls. First eval, 50 query embeddings, 8.0 seconds. Second eval, 0.17 seconds and zero
  calls. So re-scoring this rung needs no credential at all, which is what makes the
  numbers above checkable by someone who does not have one.
- The test suite would have started spending money. Ingest now embeds every chunk, its
  default backend is the real API, and five existing tests call `run_ingest` on the real
  corpus. `tests/conftest.py` sets `FINRAG_EMBEDDER=fake` at import, with a `setdefault`
  so a deliberate run against Vertex can still override it. Set in conftest rather than a
  fixture because `run_ingest` reads it through `Settings` at call time, in several files.
- The R1 verifier's punch list (`docs/followups.md`) caught a latent bug in `bm25.py`
  that `dense.py` was about to copy: the empty-corpus shortcut sat in front of the
  row-order guard, so `DenseRetriever([], dense_path=stale)` would have accepted any
  saved matrix without a word. The two branches are reordered here and a test pins it.
  The `bm25.py` version is left for integration, as that list asks.
- One test outside this rung's files had to change, the same way R1 had to touch
  `test_naive.py`. `test_bm25.py` asserted `available_modes() == ["bm25", "naive"]`, which
  every new mode breaks. It now asserts only that `bm25` is still listed, and the full
  list is pinned in `test_dense.py`, which owns the newest mode.
- `.env` had a `VERTEX_EMBEDDING_MODEL` variable carried over from the old repo, and
  reading it would have let a file copy silently change the model the gold numbers were
  measured with. The model is a constant in `config.py`, overridable by our own
  `FINRAG_VERTEX_EMBED_MODEL`, and `.env` supplies only the project and the location.
- The `.pth` hidden-flag problem is unchanged, and now bites between commands rather than
  once per session: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` was needed
  again after each `pytest` run before `finrag` would import. Every CLI command in this
  rung was run that way.

## R3: hybrid retrieval, and one Gemini call to reorder it

### Decision

Add two modes. `hybrid` fuses the BM25 and dense ranked lists with Reciprocal Rank
Fusion, k=60, over a generous depth of 30 from each side. `hybrid_rerank` takes hybrid's
top 20 and asks Gemini once, at temperature 0, for the best ordering as a JSON list of
positions, falling back to the hybrid order on any failure.

R2 ended with a clear brief for this rung. BM25 owned T5 and the questions that name a
document; dense owned T2 and lost the four near-identical quarterly reports, because it
confuses them by meaning and has no rare token left to break the tie. The measured
diagnostic was a top-ten cosine spread of 13.9 percent of the top score against BM25's
49.6 percent: dense is confident about the neighbourhood and nearly indifferent inside it.
Fusion was the obvious answer.

Fusion alone did not work, and the rerank did. That is the finding of this rung and the
table below leads with it rather than burying it.

### What was added

- `src/finrag/retrieval/hybrid.py`: `HybridRetriever(chunks, bm25, dense, depth=30)`,
  mode name `hybrid`. It takes the two sub-retrievers rather than building them, so the
  registry decides how each is loaded and the tests can hand in a stub. `rrf_weight(rank)`
  is `1 / (60 + rank)` and nothing else.
- `src/finrag/retrieval/rerank.py`: `excerpt`, `build_prompt`, `parse_order`,
  `LLMReranker` (which owns the retry loop and the fallback counters), and
  `RerankRetriever`, mode name `hybrid_rerank`. The wrapper asks its inner retriever for
  `max(k, 20)` hits, reorders them, then cuts to `k`.
- `src/finrag/llm/vertex.py`: `VertexLLM`, the first real implementation of the
  `LLMBackend` protocol `fake.py` has been satisfying since R0, plus `build_llm(settings)`.
  Shaped like `VertexEmbedder`: the client is built on first use so importing the module
  needs no credentials, and one call is one call, because retries and fallbacks belong to
  the caller that knows what a failure means.
- `config.py` gains `vertex_rerank_model`, default `gemini-2.5-flash`, read from our own
  `FINRAG_VERTEX_RERANK_MODEL`. Same reasoning as `vertex_embed_model`: a copied `.env`
  must not silently change the model a published number was measured with.
- Registry and CLI wiring for both modes. `IndexBundle` gains an `llm` field alongside
  `embedder`, for the same reason it is an object and not a path: building one may open a
  Vertex client, and only the mode that needs it should pay. `MODES_NEEDING_AN_EMBEDDER`
  now covers hybrid too, since its dense half embeds the query.
- The saved report carries a `rerank` block: llm calls, fallbacks, last failure, depth.
  A rerank run where the model failed on half the questions scores like half a rerank and
  reads like a whole one; only the counter tells you which you are looking at.
- `tests/test_hybrid.py` (21 tests) and `tests/test_rerank.py` (38). All offline.

### Alternatives considered

- Weighted score fusion, `alpha * bm25 + (1 - alpha) * cosine`. Dropped. A BM25 score is
  unbounded and corpus-dependent, a cosine sits in [-1, 1], so this needs a
  normalisation and a weight, both fitted on 40 scored questions. RRF throws the scores
  away and keeps the ranks, so there is nothing to tune and nothing to overfit.
- Tuning the RRF constant. Dropped for the same reason. 60 is the published value; a
  value chosen on this gold set would be fitted to noise and would make R3's number
  incomparable with anything.
- A fusion depth of 10, matching the eval cut. Dropped. A chunk BM25 puts at rank 25 and
  dense puts at rank 2 is exactly the case fusion exists to rescue, and a depth of 10
  would throw half that evidence away before the fusion saw it. This turned out to matter:
  the reranker's single best save came from hybrid rank 14.
- A cross-encoder reranker (`bge-reranker`, `ms-marco-MiniLM`). Dropped for now. It is
  cheaper per query and needs no credential, but it drags torch and a weights download
  into a repo that has stayed numpy-only, and its scores would not be the scores the
  served API produces. Recorded as the offline comparison a later rung could run.
- Pairwise or listwise reranking with one call per passage. Dropped. Twenty calls per
  question would score better and cost twenty times as much for a rung that has to
  justify itself against free fusion.
- Letting the model answer with chunk ids instead of positions. Dropped, and this one is
  the ABAC invariant rather than a preference. The model picks numbers out of a list we
  wrote, so it structurally cannot name a chunk that was not already in the masked, fused
  top twenty. A reranker that accepted ids could be talked into naming a denied one.
- Patching up a partial ordering by appending the numbers the model left out. Dropped. A
  model that answered `[3]` would then score as a rerank of one passage plus nineteen
  untouched, and the fallback count would say everything was fine. The bar is an exact
  permutation of 1..n, so the count is honest.
- Structured output (a response schema) instead of parsing text. Deferred. The parser has
  to exist anyway for the fallback path, and it cost 12 lines.

### Tradeoff accepted

The reranker reads chunk text and sends it to Vertex. Everything it sends is already
role-masked twice over, and this rung adds a third mask inside `RerankRetriever` before
the prompt is built, with a test that asserts the denied chunk's text never appears in any
prompt. What that does not change is that the passages leave the machine at all. Every
rung so far kept text local except for embedding it; this one ships excerpts of up to
twenty confidential passages per query to a Google endpoint. That is a real change in the
shape of the system and it belongs to whoever decides whether `hybrid_rerank` is the
served default, not to the retrieval code.

Second: one Gemini call per query, about one second and roughly 2,000 input tokens. That
is a latency and cost floor no cache can remove, because the top twenty change with the
query. The embedding cache makes a repeated dense or hybrid eval free; a repeated rerank
eval costs another 50 calls.

### The full comparison, five modes, all from real runs

Section columns are the headline, as R0 ruled. `scored` is the number of questions in the
tier scored on ranking: T4 shows 6 of 10 because 4 are ambiguous probes scored on
both-sources recall, T6 shows 2 of 8 because 6 are policy probes scored on behaviour.

Section hit rate, naive / bm25 / dense / hybrid / hybrid_rerank:

| Tier | n | scored | section hit | section MRR | leakage |
|---|---|---|---|---|---|
| T1 lexical | 8 | 8 | 1.000 / 1.000 / 0.875 / 1.000 / 1.000 | 0.740 / 0.792 / 0.812 / 0.833 / 0.896 | 0 / 0 / 0 / 0 / 0 |
| T2 paraphrase | 8 | 8 | 0.125 / 0.000 / 0.875 / 0.625 / 0.875 | 0.016 / 0.000 / 0.729 / 0.202 / 0.812 | 0 / 0 / 0 / 0 / 0 |
| T3 temporal | 10 | 10 | 0.800 / 0.800 / 0.900 / 1.000 / 0.900 | 0.625 / 0.700 / 0.653 / 0.716 / 0.662 | 0 / 0 / 0 / 0 / 0 |
| T4 multi-hop | 10 | 6 | 1.000 / 1.000 / 0.833 / 1.000 / 1.000 | 0.778 / 0.917 / 0.833 / 0.736 / 0.875 | 0 / 0 / 0 / 0 / 0 |
| T5 aggregation | 6 | 6 | 0.667 / 1.000 / 1.000 / 1.000 / 1.000 | 0.352 / 0.774 / 0.889 / 0.778 / 0.771 | 0 / 0 / 0 / 0 / 0 |
| T6 ABAC | 8 | 2 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | 0.225 / 1.000 / 1.000 / 1.000 / 1.000 | 0 / 0 / 0 / 0 / 0 |
| OVERALL | 50 | 40 | 0.725 / 0.750 / 0.900 / 0.925 / 0.950 | 0.488 / 0.637 / 0.780 / 0.663 / 0.804 | 0 / 0 / 0 / 0 / 0 |

Doc level, same five runs, kept for the record and not read as progress:

| Tier | recall@10 | doc MRR |
|---|---|---|
| T1 | 1.000 / 1.000 / 0.875 / 1.000 / 1.000 | 0.938 / 0.854 / 0.875 / 0.900 / 1.000 |
| T2 | 0.750 / 1.000 / 1.000 / 1.000 / 1.000 | 0.625 / 0.588 / 0.938 / 0.771 / 1.000 |
| T3 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | 0.783 / 0.950 / 0.850 / 0.800 / 1.000 |
| T4 | 0.917 / 0.917 / 0.833 / 0.833 / 0.917 | 0.917 / 1.000 / 0.917 / 1.000 / 0.917 |
| T5 | 0.667 / 1.000 / 1.000 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 / 1.000 / 1.000 |
| T6 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 / 1.000 / 1.000 |
| OVERALL | 0.887 / 0.988 / 0.950 / 0.975 / 0.988 | 0.746 / 0.876 / 0.912 / 0.884 / 0.988 |

Ambiguous both-sources recall@10: 0.500 (2 of 4) naive, 0.750 (3 of 4) bm25, 0.500 (2 of
4) dense, 0.750 (3 of 4) hybrid, 0.750 (3 of 4) hybrid_rerank. Policy probes 6 of 6 in
all five runs. Leakage 0 in all five, target 0. Corpus unchanged at 318 chunks over 11
documents.

Rerank fallbacks: 0 of 50. Fifty questions, fifty Gemini calls, every reply an exact
permutation of 1..20 on the first attempt. No retries were spent and the fallback path
never ran in the real run, which is why the offline tests have to prove it instead.

### Which rung has paid for itself

Overall section MRR, the headline: 0.488 naive, 0.637 bm25, 0.780 dense, 0.663 hybrid,
0.804 hybrid_rerank.

- R1 (bm25) paid: +0.149 over the floor, for a numpy index and no credential.
- R2 (dense) paid: +0.143 over bm25, and it is the rung that moved T2 from 0.000 to
  0.729. Nothing lexical could have.
- R3 as fusion alone did NOT pay. Hybrid is 0.117 BELOW dense on section MRR while
  gaining 0.025 on section hit rate. It finds the right section slightly more often and
  ranks it markedly worse.
- R3 as fusion plus rerank paid, but thinly on the headline: 0.804 against dense's 0.780,
  +0.024, for one API call per query. It is a wide win everywhere else: best section hit
  rate of any mode (0.950), best doc MRR (0.988), and 6 of the 8 T2 paraphrase questions
  now sit at rank 1 while also keeping the two quarter-collision questions dense loses.

So the honest summary is that R3 is one rung with two halves and only the pair is worth
serving. Fusion buys reach, the reranker converts reach into rank, and either alone is
worse than R2.

### What moved, and why

- Hybrid's T2 collapse is the mechanism worth understanding, because it is not the one we
  guessed. It is not tie dilution: at depth 30 the two lists overlap heavily, 454 of 500
  top-ten slots were found by both retrievers, and only 19 of 500 slots involved a
  duplicate fused score. It is that RRF punishes a strong single-list opinion. T2-07
  ("during the July to September stretch, how many people joined the point scheme") has
  its gold section at dense rank 1 and BM25 does not find it in 30 tries. RRF scores it
  1/61 = 0.0164, while nine chunks that both lists placed around rank 5 score 0.027 to
  0.031. The gold section falls to hybrid rank 14 and out of the scored top ten: section
  RR 1.000 under dense, 0.000 under hybrid. T2-02, T2-03, T2-06 and T2-08 are the same
  shape, all 1.000 down to between 0.250 and 0.500. Dense's rank-1 chunk survives as
  hybrid's rank-1 chunk on only 31 of 50 questions.
- Hybrid earns its keep on exactly the two questions R2 named as the largest remaining
  problem, and nowhere else. T1-08 ("according to the 2024 annual marketing report, what
  was CAC") was 0.000 under dense, which returned ten quarterly sections in a 0.035 score
  band; BM25 puts the annual report at rank 3, so fusion lifts it to hybrid rank 6 and
  0.167. T4-01 is the same story, 0.000 to 0.167. Also T3-01, 0.000 to 0.125. Small
  numbers, but they are moves from nothing, and the reranker turns both of the first two
  into 1.000.
- The reranker's best save is the question no earlier mode found at all. T2-04 ("which
  three nations were named as fresh territory in the April to June plan") scored 0.000
  under naive, bm25, dense and hybrid. Its gold section sits at BM25 rank 28 and dense
  rank 26, so hybrid rank 14, inside the top twenty the reranker reads and outside the
  top ten the eval scores. The model put it at position 2. That single question is the
  whole argument for reranking a window wider than the cut.
- T2 under rerank: 0.202 to 0.812 section MRR. Six of the eight questions at rank 1,
  T2-04 at rank 2, and only T2-01 still at zero. T2-01 asks about "buyer attrition during
  the festive stretch", and its gold section lands at hybrid rank 21, one slot outside the
  window the reranker reads, because BM25 never finds it and dense's rank 3 is not enough
  on its own. So this is not a rerank failure; it is the same single-list penalty, and it
  cost us the question by one position.
- T6 holds at section MRR 1.000, leakage 0, 6 of 6 policy probes, across both new modes.
  That is what a new mode should do to the ABAC tier: nothing.

### Surprises

- The reranker prefers prose to tables, and that is where all six of its regressions come
  from. T3-10 asks for the Q3 conversion rate; hybrid has
  `marketing_report_q3_2024#performance-analysis` at rank 1 and the model demotes it to
  position 4 behind `campaign-highlights--digital-marketing`. T4-02 asks which platforms
  the quarterly software spend pays for; hybrid has the Q1 and Q4 expense breakdowns at
  ranks 1 and 2 and the model puts three paragraphs of the narrative financial summary
  above them. T1-01 and T3-07 both want `q1-strategic-objectives`, a terse target list,
  and both drop. The likely cause is our own prompt: 400 characters of a table-shaped
  section is column headers and part of one row, while 400 characters of a narrative
  section reads like an answer. So the reranker is partly judging prose style rather than
  content, and a cheap experiment for a later rung is a longer excerpt for `aggregate` and
  table-shaped chunks.
- Two of those regressions are partly a gold-set limit rather than a model error. T4-02
  and T5-05 name three and four gold sections respectively, and the model's first pick is
  a passage a person would also reach for. We are not adjusting the labels to suit the
  numbers; recording it because a reader comparing 0.771 to 0.889 on T5 deserves to know
  the shape of the disagreement.
- Fusion depth mattered more than the fusion constant. Nothing in the RRF arithmetic
  changed between depth 10 and depth 30, but the reranker's window is filled from it, and
  T2-04's save came from hybrid rank 14. Had we matched the depth to the eval cut, the one
  question that separates this rung from R2 on T2 would have been invisible.
- The R1 verifier's forward note landed exactly as written: "R3's reranker inherits a
  noisy top-30 and should know why." It does. Function words hold top-five slots on the
  BM25 side because stopwords are deliberately still off, and that noise is now the
  reranker's input rather than the eval's output. It held or improved 34 of the 40 scored
  questions and lost ground on 6, without being told anything about the noise.
- `gemini-2.0-flash`, the model name sitting in `config.py` since R0 and in the brief for
  this rung, returns 404 on this project: "Publisher model ... was not found or your
  project does not have access". `gemini-2.5-flash` works. Verified with a one-line call
  before any code was written, which is the only reason it cost a minute instead of a
  debugging session. The rerank model is now its own setting with its own default, and
  `Settings.vertex_model` is left alone because nothing reads it yet.
- Gemini 2.5 thinks by default, and thinking is billed against `max_output_tokens`. The
  first test call spent several seconds and returned nothing parseable. With
  `thinking_config(thinking_budget=0)` the same call takes about one second and returns
  `[2, 1, 3]`. For a task whose entire output is a permutation of twenty numbers there is
  nothing for a thinking budget to buy.
- The `{"order": [1, 2, 3]}` case in the parser tests failed on the first run and the
  right fix was to move it from the rejected list to the accepted one. Taking the text
  between the first `[` and the last `]` means an array wrapped in an object parses, and
  that is the behaviour we want: the model told us the ordering, and the permutation check
  still has to pass. Written up as an intended test rather than left as an accident.
- One test outside this rung's files had to change, the third rung in a row to hit this.
  `test_dense.py` pinned `available_modes() == ["bm25", "dense", "naive"]`. It now asserts
  only that `dense` is listed, and the full list lives in `test_rerank.py`, which owns the
  newest mode. R1 wrote the same note about `test_naive.py`, R2 about `test_bm25.py`.
- The `.pth` hidden-flag problem is unchanged. Every CLI command in this rung ran as
  `PYTHONPATH=src uv run --no-sync python -m finrag.cli ...`, which sidesteps the console
  script entirely and needs no `chflags` at all. Worth folding into `dev.sh`.

## R4: the leakage sweep

### Decision

Add `run_leakage_sweep()` and a `finrag sweep` command that asks EVERY gold question in
EVERY policy role against EVERY retrieval mode, then re-decides every returned chunk with
`PolicyEngine.replay`. 50 questions x 6 roles x 5 modes = 1,500 cells. The command exits
non-zero on a single violation, so it is a gate rather than a report.

The reason this rung exists is a hole in the harness, and it is worth naming plainly.
`EvalHarness` asks each gold question in the ONE role that question was written for. That
is correct for scoring and useless for proving the mask: no question in the gold set is
written for `employee`, so a retriever that leaked to `employee` alone would have scored a
clean run through all of R0 to R3. Four rungs reported "leakage 0" from a check that never
asked five sixths of the questions.

### What was added

- `src/finrag/eval/sweep.py`: `run_leakage_sweep(retrievers, policy, gold, roles, k)`,
  which returns the report; `build_offline_retrievers(settings)`, which builds all five
  modes over one shared `IndexBundle`; `sweep_passed(report)`; `breakdown_lines(report)`
  for the console table; and `FallbackOnlyLLM`.
- `FallbackOnlyLLM` is a generation backend whose `generate` always raises. Not a test
  mock: the sweep uses it on purpose, so `LLMReranker` counts a failure and returns its
  input order unchanged. The sweep is about the mask, not about ranking quality, and every
  mask in `RerankRetriever.search` still runs on the fallback path.
- CLI: `finrag sweep [--k 10] [--rung r4] [--modes ...]`, writing
  `evals/results/r4_sweep_<stamp>.json` through the existing `write_report`, so the sweep
  report is named by the same rule as every eval report. `--modes` exists so the test suite
  can gate the real corpus on one mode in a second instead of five in ten.
- `tests/test_sweep.py` (14 tests) and three more in `tests/test_cli.py`. The first thing
  in `test_sweep.py` is `LeakyRetriever`, which returns every chunk to every role with no
  mask at all. A sweep test suite that only ran the real retrievers could not tell "no leak
  happened" from "the check does nothing".

### What the run says

1,500 cells, 14,908 hits inspected, 0 violations. Cells are `hits/violations`:

| role | bm25 | dense | hybrid | hybrid_rerank | naive | hits | leaks |
|---|---|---|---|---|---|---|---|
| finance | 500/0 | 500/0 | 500/0 | 500/0 | 500/0 | 2500 | 0 |
| marketing | 500/0 | 500/0 | 500/0 | 500/0 | 500/0 | 2500 | 0 |
| hr | 482/0 | 500/0 | 500/0 | 500/0 | 482/0 | 2464 | 0 |
| engineering | 490/0 | 500/0 | 500/0 | 500/0 | 490/0 | 2480 | 0 |
| c_level | 500/0 | 500/0 | 500/0 | 500/0 | 500/0 | 2500 | 0 |
| employee | 482/0 | 500/0 | 500/0 | 500/0 | 482/0 | 2464 | 0 |
| ALL | 2954/0 | 3000/0 | 3000/0 | 3000/0 | 2954/0 | 14908 | 0 |

No cell was empty, which is the number that makes the zeros mean something: every one of
the 1,500 combinations returned hits, so every one of them had something to leak. Corpus
unchanged at 318 chunks over 11 documents.

Two readings of the table beyond the headline. Dense and its two descendants return
exactly `k` for every query because cosine search has no score floor, while bm25 and naive
return only chunks with a non-zero term score, so `hr` and `employee` fall short of 500 on
those two columns. And `employee` seeing 2,464 hits at all is the general corpus doing its
job: employee is permitted `owner_dept == general` and nothing else.

### The hr_aggregate_mode tradeoff

`policy.yaml` ships `hr_aggregate_mode: disabled`, so the five HR aggregate chunks are
visible to `hr` and `c_level` only. Rule 7 would open them to the other four roles if the
flag were enabled, and that flag is exactly the kind of thing that gets flipped in a
config change nobody reviews as a security change.

The sweep is the evidence for the shipped setting. Counting doc types across the same
1,500 cells: aggregates reached `hr` 423 times and `c_level` 101 times, and `finance`,
`marketing`, `engineering` and `employee` zero times each. The 100 HR CSV rows behave the
same way, 522 hits to `hr` and 95 to `c_level`, zero to everyone else. Since replay denies
an aggregate to a non-HR role under the current flag, 0 violations across those cells is
the same statement as "no other role saw one".

What the sweep does NOT prove is the enabled case. Flip the flag and the same sweep would
report 0 violations again, because `replay` reads the same flag the mask was built from,
and both would agree that non-HR roles may now have safe summaries. That is the tradeoff
being accepted: the sweep proves the mask matches the policy, not that the policy is the
one we wanted. Whoever enables `hr_aggregate_mode` is making a policy decision that no
amount of sweeping will second-guess, and they should re-read rule 7 rather than the sweep
output. A later rung could pin the shipped value with an assertion, which would turn a
silent config flip into a failing test.

### Alternatives considered

- Folding the sweep into `EvalHarness.run` as an extra pass. Dropped. The harness returns
  metrics per question in one role, and a second cross-role pass inside it would double the
  meaning of every count in the report. The sweep answers one yes-or-no question and
  deserves its own report file and its own exit code.
- Trusting `Chunk.allowed_roles` instead of replaying. Dropped, and this is the point of
  the rung. `allowed_roles` is what the retrievers mask with, so checking the mask against
  the mask proves nothing. Replay re-derives the decision from the rules, which is what
  catches a stale stamp left behind by a policy edit without a re-ingest.
- Sweeping only the T6 ABAC questions in all roles. Dropped. T6 questions are written to
  be refused, so their retrieval is the easy case. The dangerous case is an ordinary T1
  question whose answer sits in a confidential document, asked in a role that should not
  reach it.
- Running `hybrid_rerank` with the real Gemini call. Dropped. 300 cells is 300 calls for a
  check whose answer cannot depend on the ordering, since a permutation of a masked list is
  still masked. The fallback path is the one that has to be safe anyway.
- Recording every violation. Capped at 100 records; the COUNT is always exact. A broken
  retriever leaks on all 1,500 cells and would write a 15,000-record file nobody reads.

### Surprises

- `finrag sweep` on the real corpus made zero API calls and finished in a few seconds. The
  50 question texts were already in `embed_cache.npz` from the R2 and R3 runs, and the file
  mtime confirms it was not rewritten. The cache built for cost turns out to be what makes
  a 1,500-cell offline sweep possible at all.
- `cli.py` is now 333 lines, over the ~250 guideline. Roughly a third of it is console
  table formatting for two commands. Flagged rather than fixed here, since extracting a
  `cli_print.py` would touch R0 to R3 code this rung has no business editing.
- Writing `LeakyRetriever` first changed what the rest of the tests are worth. Every clean
  assertion in `test_sweep.py` now sits next to a proof that the same assertion fails when
  the mask is removed, which is the difference between a green suite and evidence.
