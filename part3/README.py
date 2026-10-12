# Part 3 — Revision Engine

Takes an **Evidence Object** from Part 2 and a **Memory Object**, decides how the memory's
confidence should change, and returns an updated Memory Object with version history.

## Files
| File | Purpose |
|---|---|
| `revision_engine.py` | The engine: `revise_memory`, `RevisionEngine`, `apply_decision` |
| `test_revision_engine.py` | 32 unit tests (update math, safeguard, Part 2 compatibility, Memory Object handling) |
| `stress_test.py` | Parameter grid over 6 evidence patterns (learning rate x deprecate threshold) |
| `integration_demo.py` | End-to-end: Part 2 `collect_evidence` -> Part 3 -> updated memory |

## Run (Python 3.9+, no dependencies)
```
python3 -m unittest test_revision_engine.py -v     # expect: Ran 32 tests ... OK
python3 stress_test.py
python3 integration_demo.py                        # from project root so it finds part2/
```

## How it plugs into Part 2
```python
from evidence_collector import collect_evidence          # Part 2
from revision_engine import RevisionEngine, apply_decision

engine = RevisionEngine()                                # lr=0.5, threshold=0.15, min_evidence=2
evidence = collect_evidence(memory, new_information)     # dict: relation = support|contradiction|neither
decision = engine.revise_memory_object(memory, evidence) # RevisionDecision (.to_dict() for schema)
memory   = apply_decision(memory, decision)              # new Memory Object, version+1, history appended
```

## Changes made to match Part 2
- Accepts Part 2's plain-dict Evidence Objects (not only the dataclass).
- Accepts `relation = "neither"` (Part 2) and `"neutral"` (original schema); both mean no change.
- Validates relation, strength range/type, and that evidence targets the right `memory_id`.
- `apply_decision` writes confidence, version, `last_updated`, history, and `status = "deprecated"`.
- Memories whose status is not `active` are never revised.

## Update rule
`new = c + learning_rate * strength * (target - c)`, target = 1 (support) or 0 (contradiction).
Deprecation needs 2 consecutive hits below 0.15, so one strong contradiction cannot delete a memory.

## Known limits to mention in the report
- Part 2's NLI strengths are softmax probabilities, so they sit at ~0.99+ for any non-"neither"
  result. Strength therefore carries little information; `learning_rate` is effectively the only knob.
- Part 2 sometimes labels unrelated information as a contradiction at strength ~1.0
  (its own test 8: "studies computer science" vs "enjoys cooking"). Part 3 cannot detect that;
  the 2-hit deprecation rule only limits the damage.
- `overwrite` is unused: Evidence Objects carry no replacement text.

## Replacement beliefs
`RevisionEngine(replace_on_deprecate=True, replacement_prior=0.5)`: when a belief is retired because of a contradiction,
the decision carries `replacement = {"content": <the statement>, "confidence": prior + lr * strength * (1 - prior)}`.
Held (not yet retired) decisions carry none. `decision.to_dict()` leaves the key out when there is no replacement, so the
original 5-field schema is unchanged. Part 1's `store.apply_decision(...)` creates the replacement belief; the object-level
`apply_decision(memory, decision)` helper in this module only updates the one memory it is given.
