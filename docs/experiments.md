# Experiments

FinRAG tests retrieval before final prose. A fluent answer can hide weak
evidence. The main question is whether the pipeline found every needed source
section and where it ranked them.

## The 50 questions

| Group | Questions | What it tests |
|---|---:|---|
| T1 direct | 8 | The question and source share clear words |
| T2 paraphrase | 8 | The meaning matches, but the words change |
| T3 time | 10 | Similar facts appear in several quarters |
| T4 multi-step | 10 | Evidence spans sections or documents, including four conflicts |
| T5 summary | 6 | The answer joins several facts or uses a safe summary chunk |
| T6 access | 8 | Six blocked requests and two allowed controls |

Forty questions expect normal retrieval. Four expect both sides of a planted
conflict. Six expect no blocked text. The evaluation keeps those jobs separate.

Each gold row names its role, expected action, correct documents, correct
sections, answer, and tags. Before a run, the harness checks those labels
against the current chunks and policy. A bad label stops the run.

## Strategy-to-question evidence map

The [architecture guide](architecture.md#how-the-design-maps-to-questions)
states what each strategy was designed to help. This table records what the
current evidence supports. Intended use and measured use are not the same.

| Group | Designed candidates | What the current evidence says | Evidence status |
|---|---|---|---|
| T1 direct | Direct query, BM25 | Dense and both semantic graphs also handled most direct questions; the keyword graph reached `0.500` section hit | Controlled for four retrievers |
| T2 paraphrase | Dense, hybrid, HyDE, entity graph | Entity graph reached `1.000` section hit and recovered T2-04; ReAct also recovered T2-04 in a separate run | Graph result controlled; agent result is a case study |
| T3 time | Exact lexical clues and a temporal fact graph | Dense reached `0.900`; the current temporal graph fell to `0.700` because quarter and metric hubs were too broad | Controlled failure |
| T4 multi-step and conflict | Hybrid, ReAct, entity or claim-source graph | Entity graph found both sources for three of four conflicts; ReAct failed T4-03 after four similar searches | Graph result controlled; agent result is a case study |
| T5 summary | Hybrid, document structure, safe summary chunks | Dense and entity graph found at least one correct section for all six; structural graph value has not been isolated | Partial evidence |
| T6 access | Policy around every strategy | All 42 pipelines returned zero policy violations across all six roles | Full safety sweep |

## Experiment coverage

The project has a full design space, but it does not pretend that every quality
comparison has been run.

| Question | Changed axis | Kept fixed | Status |
|---|---|---|---|
| Does graph structure improve retrieval? | Dense control, keyword graph, entity graph, temporal graph | Direct query, no reranker, corpus, chunks, model, roles, `k` | Controlled and saved |
| Does query agency help? | Direct, HyDE, ReAct | Not all other parts were held fixed in the saved runs | Case studies, not a causal matrix |
| Does model reranking earn its cost? | None and LLM | Only a small set of saved pipelines exists | Partial |
| Can any composed pipeline leak evidence? | All 3 × 7 × 2 pipelines and all six roles | Corpus and policy | Full sweep, zero leaks |
| Would fact nodes fix temporal retrieval? | Proposed graph schema | Not run | Next experiment |
| Would claim nodes improve conflict coverage? | Proposed graph schema | Not run | Possible later experiment |

## Metrics

- **Section hit:** Did the first `k` results contain any correct section?
- **Section reciprocal rank:** How high was the first correct section? Rank 1
  scores `1.0`; rank 2 scores `0.5`.
- **Section MRR:** Mean section reciprocal rank over scored questions.
- **Section recall@k:** What share of all correct sections appeared in the
  first `k` results?
- **All sections@k:** Did the first `k` results contain every correct section?
- **Document recall@k:** What share of correct documents appeared?
- **Both sources:** Did a conflict question return both named documents?
- **Leakage:** Did any returned chunk fail a fresh policy check?

Section hit can hide incomplete evidence. A question with three gold sections
passes section hit after finding one. Section recall and all sections make that
gap visible.

## Controlled GraphRAG comparison

These four runs hold the corpus, chunks, roles, `k`, direct query, no reranker,
and `text-embedding-005` fixed. The semantic graph retrievers also share five
dense seeds, a three-hop limit, and the same score rule. Dense is the causal
control. Between `graph_entity` and `graph_temporal`, the selected edge layer is
the intended change.

| Pipeline | Section hit | Section MRR | Section recall@10 | All sections@10 | Document recall@10 | Both sources | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct__dense__none` | 0.900 | 0.780 | 0.883 | 0.850 | 0.950 | 2/4 | 0 |
| `direct__graph_keyword__none` | 0.425 | 0.163 | 0.354 | 0.300 | 0.950 | 1/4 | 0 |
| `direct__graph_entity__none` | 0.975 | 0.792 | 0.927 | 0.850 | 0.988 | 3/4 | 0 |
| `direct__graph_temporal__none` | 0.875 | 0.770 | 0.796 | 0.700 | 0.919 | 3/4 | 0 |

The reports are:

- [`direct__dense__none`](../evals/results/graph_compare_direct__dense__none_20260901T2221.json)
- [`direct__graph_keyword__none`](../evals/results/graph_compare_direct__graph_keyword__none_20260901T2221.json)
- [`direct__graph_entity__none`](../evals/results/graph_compare_direct__graph_entity__none_20260901T2221.json)
- [`direct__graph_temporal__none`](../evals/results/graph_compare_direct__graph_temporal__none_20260901T2221.json)

Each report stores the pipeline ID and retrieval settings. The semantic graph
reports include the embedding model, selected layer, seed count, hop limit,
score rule, and edge counts. The graph contained 5,109 entity edges, 636
structural edges, and 73 temporal edges.

## Results by question group

Each cell below is `section hit / section recall@10`.

| Group | Dense | Keyword graph | Entity graph | Temporal graph |
|---|---:|---:|---:|---:|
| T1 sanity | 0.875 / 0.875 | 0.500 / 0.500 | 1.000 / 1.000 | 0.875 / 0.875 |
| T2 paraphrase | 0.875 / 0.875 | 0.125 / 0.063 | 1.000 / 1.000 | 0.875 / 0.813 |
| T3 time | 0.900 / 0.900 | 0.300 / 0.200 | 1.000 / 0.900 | 0.700 / 0.700 |
| T4 multi-step | 0.833 / 0.722 | 0.333 / 0.111 | 0.833 / 0.722 | 1.000 / 0.722 |
| T5 summary | 1.000 / 1.000 | 0.833 / 0.833 | 1.000 / 0.958 | 1.000 / 0.833 |
| T6 allowed controls | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

The T6 quality row covers the two allowed controls. All four pipelines also
passed all six blocked probes. The broader leakage sweep is a separate gate.

That gate covered all 42 combinations of three query strategies, seven
retrievers, and two rerankers. It asked all 50 questions as all six roles. The
run checked 12,600 cells and replayed policy over 117,798 hits. It found zero
violations. See the
[`graph_leakage` report](../evals/results/graph_leakage_sweep_20260901T2234.json).

## What the graph test says

The entity graph improved overall section hit, complete section recall, document
recall, and conflict coverage over dense search. It did not improve every
question. On T2-04 it changed a complete miss into a hit at rank 8. That is
useful coverage, but weak ordering.

The temporal graph did not specialise on its target group. Dense found a gold
section on 90 percent of T3 questions; the temporal graph found one on 70
percent. On T3-04, the correct Q3 section was dense seed rank 2. It kept a
direct score of `0.5`, but temporal expansion gave more than `0.5` to enough Q1
and Q2 sections to push the correct seed out of the final ten. Broad quarter
and metric links did not separate a number collision between reports.

The keyword graph is a useful failure baseline. Its exact-word entry missed
most paraphrases. Its raw path score rewarded broad graph hubs. High document
recall did not mean useful section rank.

These results do not prove that entity graphs are always better. They show that
this entity layer helped this corpus under these fixed settings, while this
temporal layer did not do the job it was built to do.

## Build and service cost

The first `finrag graph --llm` build processed 318 chunks with
`gemini-2.5-flash`. It made 318 extraction calls, parsed 2,116 triples, and
reported no extraction fallback. Repeating the same build used 318 cache hits
and zero model calls, with the same node and edge counts.

This report uses calls instead of a dollar estimate. Cloud prices change, and
the exact charge also depends on token count. Cached builds are both cheaper
and easier to reproduce.

Fake embeddings may test contracts and access rules. They are not used as
retrieval-quality evidence in this comparison.

## Separate agency experiments

Four saved pipeline runs test query planning and reranking. They do not form a
causal matrix because some rows change more than one part.

| Pipeline | Section hit | Section MRR | Leakage |
|---|---:|---:|---:|
| [`direct__bm25__none`](../evals/results/modular_smoke_direct__bm25__none_20260831T1611.json) | 0.750 | 0.637 | 0 |
| [`direct__hybrid__llm`](../evals/results/modular_real_direct__hybrid__llm_20260831T1613.json) | 0.975 | 0.797 | 0 |
| [`hyde__hybrid__none`](../evals/results/modular_real_hyde__hybrid__none_20260831T1614.json) | 0.950 | 0.704 | 0 |
| [`react__hybrid__none`](../evals/results/modular_real_react__hybrid__none_20260831T1618.json) | 0.750 | 0.729 | 0 |

ReAct made 153 model calls for 50 questions. It rescued T2-04, but on T4-03 it
spent four similar searches and returned no evidence. The model reranker made
52 calls for 50 questions because two outputs needed a retry. See the
[walkthrough](walkthrough.md) for the traces.

The old
[`r5_graphrag_20260730T0012.json`](../evals/results/r5_graphrag_20260730T0012.json)
keeps its original strategy name. It predates the current pipeline and layered
graph, so it is not part of the controlled comparison.

## Fair comparison rules

1. Keep the corpus, chunker, gold set, role, `k`, and service versions fixed.
2. Keep two pipeline parts fixed and change one.
3. For graph-layer tests, also keep entry search, seed count, hop limit, and
   score fixed.
4. Save the full pipeline ID, settings, and search trace.
5. Report calls and fallbacks beside quality.
6. Replay every returned hit through the policy.
7. Treat zero leakage as a gate, not as a quality bonus.

## Limits

- The corpus is small and made for this project.
- Simple entity normalization does not resolve aliases or pronouns.
- The temporal layer connects broad quarter and metric labels. The failed T3
  result shows that its schema needs another design, not a stronger claim.
- Extraction quality is not hand-scored as a separate task.
- Retrieval scores do not grade final answer wording.
- The four agent runs do not cover the full strategy matrix.
