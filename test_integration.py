"""Cross-part test: Part 1 + Part 2 + Part 3 together. Run: python3 test_integration.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
for part in ("part1", "part2", "part3"):
    sys.path.insert(0, os.path.join(ROOT, part))

from memory_store import MemoryStore
from similarity import jaccard_similarity
from evidence_collector import collect_evidence
from revision_engine import RevisionEngine

OLD = "The user prefers Python over Java"
NEW = "The user now prefers Java over Python"


def feed(store, engine, statements, gate=0.25):
    for text in statements:
        hits = store.find_memories(text, limit=1)
        if not hits:
            continue
        ev = collect_evidence(hits[0], text, backend="mock", similarity=jaccard_similarity, min_relevance=gate)
        store.apply_decision(engine.revise_memory_object(hits[0], ev).to_dict())


class TestReplacementEndToEnd(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.old = self.store.add_memory(OLD, "t", 0.82)["memory_id"]

    def test_four_contradictions_retire_old_and_create_replacement(self):
        feed(self.store, RevisionEngine(), [NEW] * 4)
        old = self.store.get_memory(self.old)
        self.assertEqual(old["status"], "deprecated")
        new = self.store.get_memory(old["superseded_by"])
        self.assertEqual((new["content"], new["status"], new["replaces"]), (NEW, "active", [self.old]))
        self.assertEqual(new["confidence"], round(0.5 + 0.5 * 0.95 * 0.5, 3))

    def test_next_identical_statement_reinforces_the_replacement(self):
        eng = RevisionEngine()
        feed(self.store, eng, [NEW] * 4)
        new_id = self.store.get_memory(self.old)["superseded_by"]
        before = self.store.get_memory(new_id)["confidence"]
        feed(self.store, eng, [NEW])
        after = self.store.get_memory(new_id)
        self.assertGreater(after["confidence"], before)
        self.assertEqual(after["version"], 2)

    def test_replacement_off_leaves_nothing_behind(self):
        feed(self.store, RevisionEngine(replace_on_deprecate=False), [NEW] * 4)
        self.assertEqual(self.store.get_memory(self.old)["status"], "deprecated")
        self.assertEqual(len(self.store), 1)

    def test_one_or_two_contradictions_do_not_retire_or_replace(self):
        feed(self.store, RevisionEngine(), [NEW] * 3)
        self.assertEqual(self.store.get_memory(self.old)["status"], "active")
        self.assertEqual(len(self.store), 1)

    def test_unrelated_statements_never_create_beliefs(self):
        feed(self.store, RevisionEngine(), ["The user enjoys cooking", "It rained today"] * 5)
        self.assertEqual((len(self.store), self.store.get_memory(self.old)["confidence"]), (1, 0.82))

    def test_lineage_reads_old_to_new(self):
        feed(self.store, RevisionEngine(), [NEW] * 4)
        chain = self.store.get_lineage(self.old)
        self.assertEqual([m["content"] for m in chain], [OLD, NEW])


if __name__ == "__main__":
    unittest.main(verbosity=2)
