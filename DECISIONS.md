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
