# Adaptive Memory Evolution through Evidence-Based Revision

```
new statement
   |  Part 1  find the most related memory           part1/memory_store.py
   v
   |  Part 2  classify: support / contradiction / neither + strength   part2/ (NLI model)
   v
   |  Part 3  decide how confidence changes          part3/revision_engine.py
   v
   |  Part 1  apply it (version +1, history kept)
```
Part 4 (orchestration + evaluation) is not included yet; `main.py` is the minimal pipeline it will build on.

## 1. Run it locally
```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # pulls in torch; ~2 GB the first time

python3 run_tests.py                # all unit tests (no model needed)
python3 main.py --mock              # whole pipeline with a tiny rule-based stand-in classifier
python3 main.py                     # whole pipeline with the real NLI model (downloads ~570 MB once)
python3 main.py --similarity embed  # match + gate with sentence embeddings (downloads ~90 MB once)
python3 main.py --min-relevance 0.4 # change how related a statement must be before the classifier runs
```

## 2. Run it in Google Colab (what Part 2 already uses)
```
# upload adaptive_memory.zip to the Colab file panel, then:
!unzip -q adaptive_memory.zip
%cd adaptive_memory
!pip -q install sentence-transformers
!python3 run_tests.py
!python3 main.py
```

## What you should see
Mock run: `support` nudges confidence up, `neither` (including gated, unrelated statements) leaves it unchanged,
and repeated `contradiction` drops it step by step (0.905 -> 0.475 -> 0.249 -> 0.131). The belief is retired only on
a second consecutive hit below 0.15, and at that moment a **replacement belief** is created from the contradicting
statement (confidence 0.738). The next identical statement reinforces the replacement (0.862) instead of the retired
belief. The final store is saved to `memory_state.json`.
With the real model, strengths come out near 0.99 for any support/contradiction, so changes are bigger per step.

## Relevance gate and matching (new)
`part1/similarity.py` scores how related a statement is to a memory (0-1): `words` (stopword-filtered overlap,
offline) or `embed` (sentence embeddings). It is used twice:
1. **Routing:** `MemoryStore.find_memories` picks the memory a statement is about.
2. **Gate:** `collect_evidence(..., similarity=sim, min_relevance=t)` skips the classifier when relevance < t and
   returns `neither`, so unrelated text can no longer become a contradiction. Gated evidence carries
   `relevance` and `gated`; Part 3 ignores these extra fields.

Default thresholds (`words` 0.25, `embed` 0.35) are starting points: tune them on your synthetic dataset.
The `embed` path has not been run end to end in the development sandbox (no model download there).

## Replacement beliefs (new)
Before, a contradicted belief could be retired but nothing took its place: the system could forget but not learn the
correction. Now, when the Revision Engine retires a belief because of a contradiction, it proposes a replacement
(`decision.replacement = {content, confidence}`) and the Memory Store creates it:
- **Content** is the contradicting statement as written.
- **Starting confidence** is a neutral prior (0.5) nudged toward 1 by that same evidence, because evidence against
  the old belief is evidence for the new one (`replacement_prior`, default 0.5).
- **Links:** the old belief gets `superseded_by`, the new one gets `replaces`; `get_lineage(id)` returns the chain.
  The old belief stays in the store as `deprecated`, so the audit trail is kept.
- **No duplicates:** if an active belief is already >= 0.8 similar, it is reused and linked instead.
- **Ablation:** `python3 main.py --mock --no-replace` (or `RevisionEngine(replace_on_deprecate=False)`) turns it off,
  which gives Part 4 a with/without comparison.

## Layout
| Folder | Owner | Key entry point |
|---|---|---|
| `part1/` | Memory Store | `MemoryStore` |
| `part2/` | Evidence Collector | `collect_evidence(memory, new_information, backend="nli")` |
| `part3/` | Revision Engine | `RevisionEngine.revise_memory_object(memory, evidence)` |

## Known limits (mention in the report)
- The relevance gate reduces false contradictions but cannot catch a related statement that the model still
  misjudges; thresholds need tuning, and gating too hard hides real evidence.
- NLI strengths are softmax probabilities near 0.99, so `learning_rate` is effectively the only tuning knob
  (calibration is a possible next step).
- A replacement's content is the raw statement; rewriting it into a clean canonical belief (e.g. with an LLM) is a next step.
- `--mock` is a smoke-test stand-in, not an evaluation baseline.
