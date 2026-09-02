# Walkthrough

These traces show why I kept both question-level evidence and aggregate scores.
An average can hide the choice that helped or failed.

ReAct means Reason and Act. In FinRAG it plans searches and selects evidence. A
shared step writes the answer later.

## A query-agent win: T2-04

The question asks:

> Which three nations were named as fresh territory in the April to June plan?

The source says Q2 and calls the work a European expansion. The question avoids
both phrases. This is a paraphrase test.

Direct BM25 searched with the full question. It returned ten sections and found
the correct document at document rank 5, but none of its returned chunks matched
the correct section. Its section score was zero.

ReAct made two searches:

1. `April to June plan fresh territory nations`
2. `Q2 2024 plan European Expansion nations`

The first search returned six new sections. The second returned five. The
planner selected one section from the Q2 marketing report and put it at rank 1.
The answer in that source is Indonesia, Thailand, and Vietnam.

This run used three model calls: two search choices and one final evidence
choice. It found the right wording change. The result does not prove that every
paraphrase needs an agent. Direct hybrid search with model reranking also found
the section, at rank 2. It shows that this agent made one useful choice on this
question.

Evidence:

- [ReAct report row](../evals/results/modular_real_react__hybrid__none_20260831T1618.json)
- [Direct BM25 report row](../evals/results/modular_smoke_direct__bm25__none_20260831T1611.json)
- [Gold question](../evals/gold/gold_set.json)

## A query-agent failure: T4-03

The question asks:

> Finance flagged in Q2 2024 that influencer marketing might underperform
> against expected ROI. What ROI did marketing report for that quarter, and
> what was the target?

The answer needs two parts of the Q2 marketing report and the finance risk note.
Marketing reported 2.3x against a 2.4x target.

ReAct made four searches:

1. `influencer marketing ROI Q2 2024`
2. `influencer marketing reported ROI Q2 2024`
3. `influencer marketing target ROI Q2 2024`
4. `influencer marketing actual ROI Q2 2024`

The searches changed one word at a time. They did not break the task into its
separate facts. The planner saw eight unique sections, used five model calls,
hit its search limit, and returned no evidence. Its section score was zero.

Direct BM25 found a correct section at rank 2 with one search. Direct hybrid
search with model reranking put a correct section at rank 1.

This is the stronger lesson. The agent had more control and spent more calls,
but its search plan covered less of the question. A better next test would make
the planner state which fact is still missing before it may search again.

Evidence:

- [ReAct report row](../evals/results/modular_real_react__hybrid__none_20260831T1618.json)
- [Direct hybrid and rerank report row](../evals/results/modular_real_direct__hybrid__llm_20260831T1613.json)
- [Direct BM25 report row](../evals/results/modular_smoke_direct__bm25__none_20260831T1611.json)
- [Gold question](../evals/gold/gold_set.json)

## An entity-graph win: T2-04

The controlled graph test uses the same T2-04 question, but it changes only the
retriever. The query is direct and there is no reranker.

Dense search did not return the Q2 marketing overview in its first ten results.
The first five dense results became graph seeds. `graph_entity` followed only
entity and claim links from those seeds. It brought the missing overview back
at rank 8, for a section reciprocal rank of `0.125`.

The hit score was `0.633`. It came from four distinct seeds at ranks 1, 2, 4,
and 5. Three shortest paths had two hops and one had three. The section was not
a direct seed, so its whole score came from graph links.

This is a real retrieval gain: section recall changed from zero to one. It is
also a modest gain. Rank 8 may be too low when an answer uses only a few
sections. The graph recovered evidence better than it ordered evidence.

Evidence:

- [Dense report row](../evals/results/graph_compare_direct__dense__none_20260901T2221.json)
- [Entity graph report row](../evals/results/graph_compare_direct__graph_entity__none_20260901T2221.json)
- [Gold question](../evals/gold/gold_set.json)

## A temporal-graph failure: T3-04

The question asks:

> In Q3 2024, which activity produced 500,000 impressions, and how many
> influencers took part?

The same number appears in nearby quarter reports. The correct Q3 section says
influencer marketing used 25 regional influencers.

Dense search put that section at rank 2. It therefore entered
`graph_temporal` as seed 2 with a direct score of `0.5`.

The temporal walk linked seeds through broad quarter and metric nodes. Several
Q2 sections received contributions from more than one seed and scored above
`0.5`. They pushed the correct Q3 seed out of the first ten results. The final
list contained nine Q2 sections and one Q1 section. Section recall fell from one
to zero.

This failure is more useful than a claim that temporal structure should help.
The graph knew quarters and metrics, but its edges did not express which number
belonged to which activity. The score rewarded shared quarter context instead
of the fact asked for.

Evidence:

- [Dense report row](../evals/results/graph_compare_direct__dense__none_20260901T2221.json)
- [Temporal graph report row](../evals/results/graph_compare_direct__graph_temporal__none_20260901T2221.json)
- [Gold question](../evals/gold/gold_set.json)

## The judgment

The query-agent win and failure come from the same agent and hybrid retriever.
That is why I would not route every question through ReAct.

The current evidence supports a narrower use: try planned search when wording
is far from the source or when one fixed query is unlikely to cover the task.
Keep a simpler path for direct questions. Record the trace either way, because
the number of searches does not explain whether they were different enough to
help.

The graph traces support the same rule. I would keep the entity layer as an
experiment because it improved coverage. I would revise the temporal schema
before adding more graph methods. More links are useful only when they preserve
the relation the question needs.
