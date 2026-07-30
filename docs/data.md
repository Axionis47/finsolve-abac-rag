# The data

Everything the system knows lives in `data/corpus/`, and everything it is
tested on lives in `evals/gold/`. This page walks through both.

## The corpus

Ten files across five department folders. The folder is the owner, and
ownership drives access.

| folder | files | content |
|---|---|---|
| `engineering/` | 1 | the engineering master document: architecture, DR plan, security, processes |
| `finance/` | 2 | an annual summary and a quarterly report with per-quarter expenses, risks, cash flow |
| `marketing/` | 5 | four quarterly reports plus an annual report: campaigns, spend, CAC, churn, conversion |
| `hr/` | 1 | `hr_data.csv`, 100 employee rows: name, role, department, location, salary, leave, contact |
| `general/` | 1 | the employee handbook: policies, leave, payroll, benefits |

Roles: `finance`, `marketing`, `hr`, `engineering`, `c_level`, `employee`. A
role sees its own folder plus `general`; `c_level` sees everything; `employee`
sees only `general`. The policy file states this as rules, deny by default,
and the engine decides per chunk and per action.

## From files to 318 chunks

`finrag ingest` cuts the corpus into chunks and stamps each one:

| kind | count | one chunk is |
|---|---|---|
| `md` section | 213 | one heading's text, split further only past a token ceiling |
| `csv_row` | 100 | one HR employee |
| `aggregate` | 5 | a computed HR table, such as average salary by department |

Chunk ids are stable and readable: `doc_id#section-slug`, so
`marketing/marketing_report_q3_2024#campaign-highlights--latin-american-expansion`
names one section of one report. Stable ids make eval labels durable and diffs
legible.

One chunk as stored (text shortened):

```json
{
  "chunk_id": "marketing/marketing_report_q3_2024#campaign-highlights--latin-american-expansion",
  "doc_id": "marketing/marketing_report_q3_2024",
  "section_path": "Comprehensive Marketing Report - Q3 2024 > Campaign Highlights > Latin American Expansion",
  "owner_dept": "marketing",
  "doc_type": "md",
  "sensitivity": "internal",
  "quarter": "Q3-2024",
  "allowed_roles": ["marketing", "c_level"],
  "payload": {},
  "text": "Comprehensive Marketing Report - Q3 2024 > ... Onboarded 15 merchants, contributing $1 million in revenue ..."
}
```

`allowed_roles` is a convenience stamp written at ingest. Nothing trusts it:
retrieval recomputes visibility from the rules, and the eval harness re-judges
every returned hit from the rules again. The stamp exists so a drifted rule
shows up as a test failure, not as quiet misbehavior.

Per-department chunk counts, which explain why the eval scores section rank
rather than document recall: after the mask, finance sees 82 chunks of 318,
employee sees 56, so picking the right document is nearly free and picking the
right section is the actual test.

| owner | chunks |
|---|---|
| hr | 105 |
| engineering | 81 |
| general | 56 |
| marketing | 50 |
| finance | 26 |

## PII handling in the HR rows

An HR row's searchable text carries only what a person needs to find the row:
name, job title, department, location. Salary, leave balance, attendance and
contact details live in a `payload` dict that retrieval never returns. Reading
the payload is a separate policy action, `view_row`, permitted to `hr` and
`c_level` only.

The effect shows in the smoke receipts: HR asks where Aadhya Patel works and
gets a cited answer from her row; HR asks her salary and the system refuses,
because the salary never entered any text the model can read. Two layers, two
different questions: may this role find the row, and may it open the row.

The five aggregate chunks (salary by department, headcount by location, and so
on) exist so honest statistics survive this split. They are visible to `hr`
and `c_level` only, and the sweep proves no other role ever receives one.

## The planted contradictions

The corpus contradicts itself in four places, on purpose:

| fact | one document says | another says |
|---|---|---|
| Q2 2024 marketing spend | $550M (finance, vendor spend) | $2.5M (marketing report) |
| company founding year | 2018 (engineering doc) | 2016 (employee handbook) |
| customer acquisition | CAC $150 (annual marketing report) | CPA $13.33 (Q1 report) |
| vendor services total | $510M (finance) | $30M (marketing) |

Four gold questions target these pairs. They are scored on whether a mode
surfaces BOTH sides, because an answer built on one side of a contradiction is
confidently wrong. No mode ever surfaced both sides of any pair; the best
score is 3 of 4 pairs found by document, and the build log discusses why the
agent, built for exactly this, did worst here.

## The gold set

`evals/gold/gold_set.json`: 50 questions in 6 tiers, every one written against
verified corpus facts and real chunk ids.

| tier | n | what it isolates |
|---|---|---|
| T1 direct | 8 | plain word-match retrieval |
| T2 paraphrase | 8 | meaning without shared words ("buyer attrition" for "churn") |
| T3 temporal | 10 | the right quarter when four quarters look alike |
| T4 multi-hop | 10 | answers spread across two documents, including the 4 contradiction probes |
| T5 aggregation | 6 | computed facts, mostly from the HR aggregates |
| T6 access probes | 8 | questions whose correct outcome is refusal, plus paired controls that must be answered |

Each entry carries the question, the asking role, the expected behavior
(answer, refuse, or no answer), the gold documents and sections, the gold
answer text, and its distractors. `finrag eval` validates the whole set before
scoring: every referenced document must exist in the built chunks, every gold
section must match a real chunk id, tier counts must hold, and the asking role
must be allowed to see its gold chunks (except in T6, where denial is the
point). A drifted label fails the run instead of quietly bending a score.
