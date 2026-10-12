# Part 1 - Memory Store & Data Model

Holds every belief as a Memory Object, tracks full version history, and gives Parts 2-4 a safe API.

## Files
| File | Purpose |
|---|---|
| `memory_store.py` | `MemoryStore` class + module-level `add_memory / get_memory / update_memory` |
| `test_memory_store.py` | 26 unit tests |
| `pipeline_demo.py` | Part 1 + Part 2 + Part 3 running together (run from the project root) |

## Run (Python 3.9+, no dependencies)
```
python3 -m unittest test_memory_store.py -v     # expect: Ran 26 tests ... OK
python3 pipeline_demo.py                        # from project root: needs part3/ beside part1/
```

## Memory Object (shared schema)
`memory_id, content, confidence (0-1), created_at, last_updated, source, version, history[], status, replaces[], superseded_by`
- `status`: `active` | `deprecated` | `merged`
- `history` entry: `{version, confidence, content, updated_at, reason}`; `reason` says why that version was replaced.

## API
| Call | What it does |
|---|---|
| `add_memory(content, source, confidence=0.5)` | Create (version 1, active). Ids: `mem_0001`, `mem_0002`, ... |
| `get_memory(id)` | One memory, or `None` |
| `list_memories(status="active")` | Filter by status (`None` = all) |
| `find_memories(query, limit=5)` | Rank active memories by word overlap; use to pick the target memory for new information |
| `update_memory(id, new_content, new_confidence, status, reason)` | Versioned change; old state goes to history |
| `apply_decision(decision)` | Executes Part 3's Revision Decision (`update_confidence`, `overwrite`, `deprecate`, `no_change`) |
| `get_history(id)` | Every version, oldest first, current last |
| `mark_merged(id, into_id)` | Retire a duplicate |
| `replace_memory(id, new_content, confidence, reason)` | Retire a belief and link it to its replacement (reuses a near-duplicate active belief if one exists) |
| `get_lineage(id)` | Chain of beliefs it replaced / was replaced by, oldest first |
| `save(path)` / `MemoryStore.load(path)` | JSON persistence |

## Design rules
- Every change bumps `version` and pushes the previous state onto `history`; `no_change` never bumps.
- Invalid input (bad confidence, empty content, unknown status) raises before anything is modified.
- Returned objects are copies, so callers cannot edit the store by accident.

## How it connects
```python
store = MemoryStore()
store.add_memory("The user prefers Python over Java", "log_12", 0.82)

memory   = store.find_memories(new_info, limit=1)[0]          # Part 1
evidence = collect_evidence(memory, new_info)                 # Part 2
decision = engine.revise_memory_object(memory, evidence)      # Part 3
store.apply_decision(decision.to_dict())                      # Part 1
```

## Replacement beliefs
`apply_decision` accepts an optional `replacement: {content, confidence}` on a `deprecate` decision, validates it before
changing anything, retires the old belief, and creates the linked replacement. `replaces` / `superseded_by` are link
metadata and do not bump `version`.
