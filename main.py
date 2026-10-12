"""
Adaptive Memory Evolution - full pipeline (Parts 1 + 2 + 3).

    python3 main.py --mock                  # no model, runs anywhere (smoke test)
    python3 main.py                         # real NLI classifier (needs sentence-transformers)
    python3 main.py --similarity embed      # embedding-based matching and relevance gate
    python3 main.py --no-replace            # retire beliefs without creating replacements (ablation)

Per incoming statement:
    Part 1 finds the most related memory  ->  Part 2 gates for relevance and classifies the evidence
    ->  Part 3 decides the revision        ->  Part 1 applies it (versioned); a retired belief gets a
    linked replacement built from the statement that contradicted it.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for part in ("part1", "part2", "part3"):
    sys.path.insert(0, os.path.join(ROOT, part))

from memory_store import MemoryStore            # Part 1
from similarity import get_similarity, DEFAULT_MIN_RELEVANCE
from evidence_collector import collect_evidence  # Part 2
from revision_engine import RevisionEngine       # Part 3

SEED_MEMORIES = [
    ("The user prefers Python over Java", "conversation_log_12", 0.82),
    ("The library closes at 6 PM", "library_notice", 0.80),
]

STREAM = [
    "The user still prefers Python over Java",
    "The user bought a new laptop yesterday",
    "The user enjoys cooking",
    "The user now prefers Java over Python",
    "The user now prefers Java over Python",
    "The user now prefers Java over Python",
    "The user now prefers Java over Python",   # retires the old belief and creates its replacement
    "The user now prefers Java over Python",   # now reinforces the replacement
    "The library closes at 6 PM",
]


def run(backend, stream, state_path, sim_name="words", min_relevance=None, replace=True):
    sim = get_similarity(sim_name)
    gate = DEFAULT_MIN_RELEVANCE[sim_name] if min_relevance is None else min_relevance
    store = MemoryStore(similarity=sim)
    for content, source, conf in SEED_MEMORIES:
        store.add_memory(content, source, conf)
    engine = RevisionEngine(learning_rate=0.5, deprecate_threshold=0.15, min_evidence_for_deprecation=2,
                            replace_on_deprecate=replace)

    print(f"Classifier backend: {backend} | similarity: {sim_name} | relevance gate: {gate} | "
          f"replacement beliefs: {'on' if replace else 'off'}\n")
    for info in stream:
        hits = store.find_memories(info, limit=1)               # routing = similarity ranking
        if not hits:
            print(f"'{info}'\n  -> no related active memory; skipped\n")
            continue
        memory = hits[0]
        evidence = collect_evidence(memory, info, backend=backend, similarity=sim, min_relevance=gate)
        decision = engine.revise_memory_object(memory, evidence)
        updated = store.apply_decision(decision.to_dict())
        note = f" [gated: relevance {evidence['relevance']} < {gate}]" if evidence.get("gated") else ""
        print(f"'{info}'\n  -> {memory['memory_id']} | {evidence['relation']} ({evidence['strength']}){note} | "
              f"{decision.action} | conf {memory['confidence']} -> {updated['confidence']} | "
              f"v{updated['version']} {updated['status']}")
        if updated.get("superseded_by"):
            new = store.get_memory(updated["superseded_by"])
            print(f"  -> replacement: {new['memory_id']} \"{new['content']}\" at confidence {new['confidence']}")
        print()

    print("Final state")
    for m in store.list_memories(None):
        link = (f"  replaced by {m['superseded_by']}" if m.get("superseded_by") else "") + \
               (f"  replaces {', '.join(m['replaces'])}" if m.get("replaces") else "")
        print(f"  {m['memory_id']}  {m['status']:<10} conf={m['confidence']:<6} v{m['version']}  {m['content']}{link}")
    store.save(state_path)
    print(f"\nSaved to {state_path}")
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the rule-based stand-in instead of the NLI model")
    ap.add_argument("--state", default="memory_state.json", help="where to save the final memory store")
    ap.add_argument("--similarity", choices=["words", "embed"], default="words",
                    help="how statements are matched to memories and gated: word overlap (offline) or sentence embeddings")
    ap.add_argument("--min-relevance", type=float, default=None,
                    help="statements less similar than this to a memory are treated as 'neither' (default depends on --similarity)")
    ap.add_argument("--no-replace", action="store_true", help="do not create replacement beliefs when a belief is retired")
    args = ap.parse_args()
    run("mock" if args.mock else "nli", STREAM, args.state, args.similarity, args.min_relevance, not args.no_replace)
