# Data

The data exists to test RAG choices. It is not the goal of the project.

FinSolve is a fictional company. Its corpus is small enough to inspect by hand
but varied enough to test similar quarters, changed wording, joined facts,
conflicting sources, tables, employee rows, and role limits.

## The corpus

There are ten source files:

| Folder | Files | Content |
|---|---:|---|
| `engineering` | 1 | Architecture, recovery, security, and process notes |
| `finance` | 2 | An annual summary and a four-quarter report |
| `marketing` | 5 | Four quarter reports and one annual report |
| `hr` | 1 | One CSV with 100 employee rows |
| `general` | 1 | The employee handbook |

The default ingest produces 318 chunks:

| Chunk type | Count | Use |
|---|---:|---|
| Markdown | 213 | Source sections split by headings |
| HR row | 100 | One restricted employee record per chunk |
| HR summary | 5 | Safe computed facts such as headcount and average salary |

Chunk IDs are stable and readable. A marketing section looks like:

```text
marketing/marketing_report_q3_2024#campaign-highlights--latin-american-expansion
```

Stable IDs let the gold set point to exact source sections.

## Graph inputs

The graph does not use a second corpus. It turns the same chunks into three
edge layers:

- structural links join chunks, documents, and owning departments;
- temporal links join chunks to quarter and metric labels found by fixed
  parsing rules; and
- entity links join chunks to model-extracted entities and claims.

Keeping one chunk set matters for the experiment. Dense search and the two
semantic graph retrievers start from the same text. Their comparison changes
the graph walk, not the source material.

## Roles

The policy defines six roles:

| Role | Default view |
|---|---|
| `finance` | Finance and general sections |
| `marketing` | Marketing and general sections |
| `hr` | HR and general sections |
| `engineering` | Engineering and general sections |
| `c_level` | All sections |
| `employee` | General sections only |

Rules, not this table, are the source of truth. Ingest asks the policy which
roles may retrieve each chunk and stores the result on that chunk.

## Employee data

The HR CSV is useful for testing row-level limits and summary questions. Its
search text contains only four fields: name, role, department, and location.

Email, birth date, joining date, manager, salary, leave, attendance, rating,
and review date stay out of search text. They remain in a separate payload. The
answer path does not return that payload.

The five HR summary chunks are calculated during ingest. They cover headcount,
salary, location, tenure, and role counts. This avoids asking a language model
to calculate exact totals from many employee rows.

## Planted conflicts

Four facts disagree across documents on purpose:

| Fact | First source | Second source |
|---|---|---|
| Q2 marketing spend | Finance says $550M | Marketing says $2.5M |
| Founding year | Engineering says 2018 | The handbook says 2016 |
| Customer acquisition | Annual marketing says $150 CAC | Q1 marketing says $13.33 CPA |
| Vendor services | Finance totals $510M | Marketing says $30M |

The correct retrieval behavior is to show both sources. Returning only one can
lead to a confident but incomplete answer. The four questions are scored on
whether both documents appear in the first ten hits.

## The gold set

`evals/gold/gold_set.json` contains 50 questions:

| Group | Count | Test |
|---|---:|---|
| T1 | 8 | Direct wording |
| T2 | 8 | Paraphrase |
| T3 | 10 | Time and similar quarter reports |
| T4 | 10 | Multi-step evidence and planted conflicts |
| T5 | 6 | Joined facts and summaries |
| T6 | 8 | Access denials and allowed controls |

Each entry stores the question, role, expected behavior, correct documents,
correct sections, answer, tags, and notes. Before scoring, `finrag eval` checks
that the named documents and sections still exist. It also checks that the role
may see normal gold evidence. A bad label stops the run instead of changing the
score in silence.

See [Experiments](experiments.md) for the scoring rules and saved results.
