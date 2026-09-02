# FinRAG docs

Start with the [project README](../README.md). It explains the question behind
FinRAG, the main design choices, and the results.

Use these guides when you want more detail:

- [Architecture](architecture.md) maps each question type to the strategies and
  graph shapes designed to help it, then explains the flow and model limits.
- [Experiments](experiments.md) separates design intent from measured evidence,
  then explains the test set, saved runs, and fair-comparison rules.
- [Walkthrough](walkthrough.md) follows one success and one failure for both
  ReAct and graph retrieval.
- [Security](security.md) shows each access check and the threat it blocks.
- [Data](data.md) describes the sample corpus, chunks, conflicts, and gold
  questions.
- [Running FinRAG](running.md) covers offline setup, Vertex, Neo4j, the API, the
  web app, and evaluation.
