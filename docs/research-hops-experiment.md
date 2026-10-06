# Experiment: research objectives across search hops

Status: proposed experiment, not implemented or measured. The answerer and model reranker live in the
neighboring Wiper repository; this document records the design in Thread. Inspection used Thread `74404e0`
and Wiper `5704a97`. No production strategy changes are proposed until the comparison is measured.

## Question and hypothesis

Does allocating successive searches to different research objectives produce a more complete, better
grounded answer than repeatedly searching for relevant answers, at the same search and context budget?

The unit of optimization is the **accumulated evidence set**, not the best result from the latest search.
Query generation and passage reranking should share each hop's objective. Otherwise a search can find a
valuable contradiction only for an answer-seeking reranker to push it off the visible page.

This is an application-layer experiment. Thread's retrieval and rank-fusion arithmetic can stay the same.

## Four search passes

Start with a fixed schedule so that the treatment is reproducible. Each objective concerns the original
question and the exact entities, dates and scope it names. Search objectives are instructions to the planner
and judge, not strings such as "find conflicting information" appended blindly to a BM25 query.

| Pass | Objective | Answerer's research question | Reranker's preference |
|---|---|---|---|
| 1: establish | Find direct evidence | What are the candidate answer and its supporting facts? | Passages that directly answer the original question |
| 2: challenge | Test a provisional claim | What would contradict, supersede, or materially qualify the strongest claim found so far? | Evidence of a conflicting outcome, explicit exception, changed decision, or boundary condition for that exact claim |
| 3: verify | Independently check the claim | Which primary record could establish what actually happened, or resolve the apparent disagreement? | Direct records and independent corroboration; evidence explaining differences in date, population, version or scope |
| 4: complete | Fill the most important gap | Which requested subquestion or plausible alternative remains untested? | Evidence answering the missing part, testing an alternative explanation, or resolving the remaining uncertainty |

For example, for "Why was project Atlas delayed, and who owns the recovery?": establish the reported cause;
look for records disputing that cause or changing the schedule; verify against the decision or incident
record; then search for the recovery owner if still missing. Search terms should use actual names, dates,
decision IDs and alternative causes discovered in the corpus.

The challenge pass needs a specific claim and cited evidence. If pass 1 finds none, use that slot to recover
missing direct evidence and log `no_claim_to_challenge`. Do not invent a position to contradict. Likewise,
if no credible conflict is found, record that outcome; it is not proof that the provisional answer is true.
The verify pass may strengthen or reject the provisional answer. A newer document is not automatically
authoritative: distinguish event time, effective time, publication time and the question's requested period.

## Matched comparison

Use `thread-sdk-lead` as the proposed parent, with its resolved configuration frozen in the run artifacts.
Do not compare against a different index or embedding model to estimate the effect of research objectives.
If the active Wiper reference changes before execution, choose one parent and derive every arm from it.

| Arm | Follow-up query objective | Reranking objective | Purpose |
|---|---|---|---|
| A: relevance control | Find more information answering the question | Answer relevance | Budget-matched control |
| B: queries only | Establish → challenge → verify → complete | Answer relevance | Is query diversification sufficient? |
| C: reranker only | Find more information answering the question | Establish → challenge → verify → complete | Can different selection recover useful evidence from ordinary searches? |
| D: coordinated | Establish → challenge → verify → complete | Same objective as the search | Main treatment |

Keep the original question as the answer-relevance judge target in A and B; give it the retrieval query as
context. This prevents B's judge from implicitly adopting a challenge objective just because the search
query asks about a contradiction. All arms create the same structured claim/gap state; the two switches
control which components receive objective-specific instructions. Later states can naturally diverge as
the retrieved evidence diverges. Compare final outcomes as paired runs, not as identical later-hop inputs.

Also run the unchanged parent as an operational reference. A versus the unchanged parent measures the
effect of enforcing a budget and changing evidence assembly. D versus A is the main research-objective
comparison. Do not credit the full D-versus-parent difference to objective rotation.

Freeze these limits before the pilot:

- Four searches total, **including initial retrieval**; one query variant per search. This is three
  follow-up searches, not four extra ones. Keep per-index candidate depth and fusion weights identical.
- At most 30 documents × 3 excerpts × 900 characters judged per search, matching the proposed parent's
  configured limits; show at most six documents per search. Record actual counts when fewer are available.
- Up to three line-read calls total, each capped at 6,000 characters, available equally to every arm.
- Three bounded follow-up planning rounds, each generating one search and at most one read request from
  already visible sources. Then one final synthesis call. Initial retrieval uses the original question.
- A 28,000-character final evidence packet including source labels; the same packing algorithm and final
  answer prompt in A–D. Fix model output limits, model/provider settings and planner-state limits too.

These are budget ceilings, not a claim of identical realized tokens or cost. Log actual input/output tokens,
calls, wall time and billed cost. Wiper's `max_hops` counts model rounds and can contain multiple tool calls;
it does not enforce a search-call budget. Enforce search and read limits separately in the experiment loop.
Do not silently add an extra round for malformed tool output or provider retries; record failures and costs.

## Shared evidence state and final selection

Maintain a compact ledger between passes: requested subquestions, provisional claims with source pointers,
unresolved disagreements, missing evidence, and searched queries. Store raw source excerpts separately from
model-written notes. Bound the serialized ledger to 4,000 characters in every arm and log truncation.

Each accumulated passage retains document ID, exact line range, text, originating search/pass, and local
rank. Merge overlapping ranges in the same document without losing provenance. A second passage from the
same document can still add a new fact. Count copied or quoted material as one origin when provenance is
available; document count alone is not independent corroboration.

For the first experiment, use a deterministic, shared packing rule: reserve one best-ranked unique excerpt
from each pass with results, then fill round-robin from each pass's ranked excerpts until the character cap.
Include successful line reads alongside the pass that requested them, before its remaining excerpts.
If an excerpt is already present, move to the next unique excerpt. Never reserve space for a fabricated
conflict, and do not compare raw Jev scores across calls as though they were calibrated probabilities.
Log which excerpts were dropped for space. This keeps one late search from replacing the whole evidence set.

Give the final answerer the original question and this source packet, using the same prompt in every arm:

> Answer all parts of the question using the supplied sources. Cite the documents supporting each material
> claim. Distinguish disagreement from differences in time or scope. Explain consequential unresolved
> conflicts and missing evidence. Weigh directness and source quality rather than counting documents.

Keep strategy names and objective labels out of final synthesis and evaluation. The final answerer chooses
which evidence to use; it is not required to mention every research pass or manufacture a balanced debate.

## Planner and reranker contract

Proposed per-search metadata (not an existing SDK API):

```json
{
  "research": {
    "version": 1,
    "pass": 2,
    "objective": "challenge",
    "original_question": "Why was Atlas delayed, and who owns the recovery?",
    "target_claim": "The delay was caused by vendor approval.",
    "target_sources": [{"doc_id": "example-source", "lines": [12, 18]}],
    "missing_parts": ["recovery owner"],
    "query": "Atlas vendor approval delay revised decision"
  }
}
```

Planner instruction for the challenge pass:

> Identify the strongest provisional claim supported by the current sources. Form one focused search for
> evidence that could falsify or qualify it, preserving the question's entities and scope. Use discovered
> corpus terms. Return the claim, its source pointers, and the search query. Do not assert a conflict exists.

Judge instruction for that pass:

> Which passage most usefully tests the target claim for the original question? Prefer an explicit
> counterexample, correction, exception, or superseding decision about the same entity and scope. Mere
> topical similarity or negative wording is insufficient. The target claim is provisional, not ground truth.

Keep the existing judge call count and rank-fusion weights. Change both its choice criterion and fit
criterion: "contains an answer" and "useful for this objective" are different quantities. Log objective fit
separately from answer fit, and treat the judge's choice as relative ranking, not proof of contradiction.
Use evidence-backed planner notes or offline annotation to classify conflicts, not choice probabilities.

Implementation points identified in Wiper:

- Add versioned experimental answerer and reranker components alongside
  `src/wiper/components/answering/agentic_v20.py` and `components/rerank/jev_excerpts_v1.py`.
  The former currently encourages immediate stopping once an answer appears; the latter imports an
  answer-seeking passage-choice instruction from `jev_chunks_v1.py`.
- Carry objective metadata through `AnswerEnv.parse` → `ParsedQuery.meta` → `AnswerEnv.retrieve`, rather
  than inserting planner instructions into searchable document terms. Apply the same metadata to initial
  retrieval. Include objective, target claim, prompt version and relevant ledger state in judge cache keys.
- Preserve query-dependent excerpt cutting, but ensure the answerer's page renders the exact windows the
  judge selected. Recutting around only original question terms can hide the contradictory sentence.
- Keep retries, errors, empty pages and fused-order fallbacks visible in traces; never silently mark a
  failed challenge as completed successfully. No change is needed to Thread's `rerank::fuse` arithmetic.

## Execution and evaluation

1. Validate locally with small fixtures: an explicit contradiction, an older superseded decision, two
   similarly named entities, duplicated sources, a question with no conflict, and missing evidence. Check
   metadata propagation, cache separation, line provenance, objective prompts and hard call/context limits.
2. Pilot A–D plus the unchanged parent on the same 50 seeded questions from a frozen
   `enterprise-rag@5k-q150-ht` corpus. Use the pilot to catch instrumentation defects and estimate cost;
   do not select a winner from 50 questions. Keep all unanswered/failed queries in the outcome denominator.
3. Freeze prompts, budgets and evaluation rules, then run all arms on the full 150-question rung. Mark
   pilot questions as development data and report the other 100 separately. Run three repeats per question
   and arm with randomized arm order. Record provider/model versions; report nondeterminism even at a
   fixed temperature. Group repeats by question when estimating uncertainty.
4. Advance promising results to the existing 20k/250 and 50k/350 rungs. Because these rungs overlap, report
   newly introduced questions separately; use a disjoint question set for a confirmatory holdout if available.

Primary outcome: paired final-answer correctness. Report the mean difference, question-clustered bootstrap
95% interval, and query-level wins/losses/ties for D versus A. Treat B and C comparisons as exploratory.
Also report completeness, citation support, unsupported claims, empty answers, cost and p50/p95 latency.
Freeze the judge rubric and blind it to arm identity; hand-review a balanced sample of wins and losses.

Measure the mechanism separately: newly covered required facts per pass, duplicate evidence rate,
gold-document coverage in the final packet, and whether discovered conflicts survive packing and affect
the answer appropriately. Initial retrieval nDCG alone cannot assess a later challenge pass. Existing
qrels may omit valuable counterevidence, so add a blinded, human-annotated audit of 30 questions covering
real conflicts, apparent conflicts resolved by scope/time, and no-conflict controls. Report conflict recall,
false conflict rate and resolution quality there; do not assume the benchmark labels cover contradictions.

Proposed advancement gate, to freeze before confirmatory evaluation: D improves correctness by at least
2 percentage points over A with the paired 95% interval above zero; average cost and p95 latency are each
at most 1.25× A. For groundedness, require the upper 95% bound on the increase in unsupported-claim rate
to be below 2 points. A small or inconclusive sample warrants more data, not an improvement claim.

Save resolved configs, repository revisions, dataset/query IDs, prompt hashes, per-pass requests/results,
source packets, final answers, evaluation outputs and failures. No experimental results exist yet.

If the fixed schedule works, test an adaptive follow-up: choose the next objective from the largest remaining
gap, permit early stopping, and stop when searches add no decision-relevant evidence. Keep that separate
from this first experiment so objective diversity is not confounded with a new stopping policy.
