"""Run: python3 -m unittest test_memory_store.py -v"""
import os
import tempfile
import unittest

from memory_store import MemoryStore
from similarity import jaccard_similarity, get_similarity, DEFAULT_MIN_RELEVANCE


class TestCreateRead(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore()

    def test_add_returns_full_schema(self):
        m = self.s.add_memory("The user prefers Python over Java", "log_12", 0.82)
        self.assertEqual(set(m), {"memory_id", "content", "confidence", "created_at",
                                  "last_updated", "source", "version", "history", "status",
                                  "replaces", "superseded_by"})
        self.assertEqual((m["replaces"], m["superseded_by"]), ([], None))
        self.assertEqual((m["version"], m["status"], m["history"]), (1, "active", []))

    def test_ids_are_sequential_and_unique(self):
        ids = [self.s.add_memory(f"fact {i}", "t")["memory_id"] for i in range(3)]
        self.assertEqual(ids, ["mem_0001", "mem_0002", "mem_0003"])

    def test_default_confidence(self):
        self.assertEqual(self.s.add_memory("x", "t")["confidence"], 0.5)

    def test_rejects_bad_input(self):
        for args in [("", "t", 0.5), ("   ", "t", 0.5), ("x", "t", 1.2), ("x", "t", -0.1), ("x", "t", "high"), ("x", "t", True)]:
            with self.assertRaises(ValueError):
                self.s.add_memory(*args)

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.s.get_memory("mem_9999"))

    def test_returned_objects_are_copies(self):
        m = self.s.add_memory("x", "t", 0.5)
        m["confidence"] = 0.99
        self.assertEqual(self.s.get_memory(m["memory_id"])["confidence"], 0.5)

    def test_list_filters_by_status(self):
        a = self.s.add_memory("a", "t")["memory_id"]
        self.s.add_memory("b", "t")
        self.s.update_memory(a, status="deprecated")
        self.assertEqual(len(self.s.list_memories()), 1)
        self.assertEqual(len(self.s.list_memories("deprecated")), 1)
        self.assertEqual(len(self.s.list_memories(None)), 2)

    def test_list_rejects_bad_status(self):
        with self.assertRaises(ValueError):
            self.s.list_memories("gone")


class TestUpdateAndHistory(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore()
        self.id = self.s.add_memory("The user prefers Python", "t", 0.8)["memory_id"]

    def test_update_bumps_version_and_records_history(self):
        m = self.s.update_memory(self.id, new_confidence=0.5, reason="contradiction")
        self.assertEqual(m["version"], 2)
        self.assertEqual(m["confidence"], 0.5)
        self.assertEqual(m["history"][0]["confidence"], 0.8)
        self.assertEqual(m["history"][0]["version"], 1)
        self.assertEqual(m["history"][0]["reason"], "contradiction")

    def test_content_update_preserves_old_content_in_history(self):
        m = self.s.update_memory(self.id, new_content="The user prefers Java")
        self.assertEqual(m["history"][0]["content"], "The user prefers Python")
        self.assertEqual(m["content"], "The user prefers Java")

    def test_last_updated_changes(self):
        before = self.s.get_memory(self.id)["last_updated"]
        after = self.s.update_memory(self.id, new_confidence=0.6)["last_updated"]
        self.assertGreaterEqual(after, before)

    def test_empty_update_raises_and_does_not_bump(self):
        with self.assertRaises(ValueError):
            self.s.update_memory(self.id)
        self.assertEqual(self.s.get_memory(self.id)["version"], 1)

    def test_invalid_update_values_raise_and_leave_memory_unchanged(self):
        for kw in [{"new_confidence": 2}, {"new_content": " "}, {"status": "gone"}]:
            with self.assertRaises(ValueError):
                self.s.update_memory(self.id, **kw)
        self.assertEqual(self.s.get_memory(self.id)["version"], 1)
        self.assertEqual(self.s.get_memory(self.id)["history"], [])

    def test_unknown_id_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.s.update_memory("mem_9999", new_confidence=0.5)

    def test_get_history_includes_current_version_last(self):
        self.s.update_memory(self.id, new_confidence=0.6)
        self.s.update_memory(self.id, new_confidence=0.4)
        h = self.s.get_history(self.id)
        self.assertEqual([x["version"] for x in h], [1, 2, 3])
        self.assertEqual([x["confidence"] for x in h], [0.8, 0.6, 0.4])

    def test_mark_merged(self):
        other = self.s.add_memory("The user likes Python", "t")["memory_id"]
        m = self.s.mark_merged(self.id, other)
        self.assertEqual(m["status"], "merged")
        with self.assertRaises(KeyError):
            self.s.mark_merged(self.id, "mem_9999")


class TestApplyDecision(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore()
        self.id = self.s.add_memory("The user prefers Python", "t", 0.82)["memory_id"]

    def dec(self, action, conf=None, content=None):
        return {"memory_id": self.id, "action": action, "new_confidence": conf,
                "new_content": content, "reason": "test"}

    def test_update_confidence(self):
        m = self.s.apply_decision(self.dec("update_confidence", 0.455))
        self.assertEqual((m["confidence"], m["version"], m["status"]), (0.455, 2, "active"))

    def test_deprecate_sets_status(self):
        m = self.s.apply_decision(self.dec("deprecate", 0.114))
        self.assertEqual((m["status"], m["confidence"]), ("deprecated", 0.114))

    def test_overwrite_changes_content(self):
        m = self.s.apply_decision(self.dec("overwrite", 0.6, "The user prefers Java"))
        self.assertEqual(m["content"], "The user prefers Java")

    def test_no_change_does_not_bump_version(self):
        self.assertEqual(self.s.apply_decision(self.dec("no_change"))["version"], 1)

    def test_unknown_action_and_missing_fields(self):
        with self.assertRaises(ValueError):
            self.s.apply_decision(self.dec("explode"))
        with self.assertRaises(ValueError):
            self.s.apply_decision({"action": "no_change"})

    def test_decision_for_unknown_memory(self):
        with self.assertRaises(KeyError):
            self.s.apply_decision({"memory_id": "mem_9999", "action": "no_change"})


class TestSearchAndPersistence(unittest.TestCase):
    def test_find_memories_ranks_by_overlap(self):
        s = MemoryStore()
        s.add_memory("The user prefers Python over Java", "t")
        s.add_memory("The library closes at 6 PM", "t")
        s.add_memory("The user lives in Bengaluru", "t")
        top = s.find_memories("The user now prefers Java over Python")
        self.assertEqual(top[0]["content"], "The user prefers Python over Java")

    def test_find_memories_no_overlap_is_empty(self):
        s = MemoryStore()
        s.add_memory("The library closes at 6 PM", "t")
        self.assertEqual(s.find_memories("zebra"), [])

    def test_find_skips_inactive_by_default(self):
        s = MemoryStore()
        mid = s.add_memory("The user prefers Python", "t")["memory_id"]
        s.update_memory(mid, status="deprecated")
        self.assertEqual(s.find_memories("user prefers Python"), [])
        self.assertEqual(len(s.find_memories("user prefers Python", status=None)), 1)

    def test_save_and_load_roundtrip(self):
        s = MemoryStore()
        mid = s.add_memory("fact", "t", 0.7)["memory_id"]
        s.update_memory(mid, new_confidence=0.4, reason="r")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "store.json")
            s.save(path)
            s2 = MemoryStore.load(path)
        self.assertEqual(s2.get_memory(mid), s.get_memory(mid))
        self.assertEqual(s2.add_memory("next", "t")["memory_id"], "mem_0002")  # counter restored


class TestSimilarity(unittest.TestCase):
    def test_jaccard_ignores_function_words(self):
        # "the" and "at" must not create overlap between unrelated sentences
        self.assertEqual(jaccard_similarity("The cat sat at the door", "The dog ran at the park"), 0.0)

    def test_jaccard_identical_and_empty(self):
        self.assertEqual(jaccard_similarity("user prefers python", "user prefers python"), 1.0)
        self.assertEqual(jaccard_similarity("", "anything"), 0.0)
        self.assertEqual(jaccard_similarity("the a an", "the"), 0.0)

    def test_get_similarity_words_and_unknown(self):
        self.assertIs(get_similarity("words"), jaccard_similarity)
        with self.assertRaises(ValueError):
            get_similarity("bogus")

    def test_default_thresholds_exist_for_each_scorer(self):
        self.assertEqual(set(DEFAULT_MIN_RELEVANCE), {"words", "embed"})

    def test_store_uses_custom_similarity(self):
        s = MemoryStore(similarity=lambda a, b: 1.0 if "java" in b.lower() else 0.0)
        s.add_memory("The user likes Java", "t")
        s.add_memory("The user likes Python", "t")
        self.assertEqual([m["content"] for m in s.find_memories("anything")], ["The user likes Java"])

    def test_find_memories_scored_is_sorted_with_scores(self):
        s = MemoryStore()
        s.add_memory("The user prefers Python over Java", "t")
        s.add_memory("The user lives in Bengaluru", "t")
        out = s.find_memories_scored("The user now prefers Java over Python")
        self.assertEqual(out[0][0]["content"], "The user prefers Python over Java")
        self.assertGreater(out[0][1], out[1][1])

    def test_load_accepts_similarity(self):
        s = MemoryStore()
        s.add_memory("fact one", "t")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            s.save(path)
            s2 = MemoryStore.load(path, similarity=lambda a, b: 1.0)
        self.assertEqual(len(s2.find_memories("whatever")), 1)


class TestReplacementBeliefs(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore()
        self.old = self.s.add_memory("The user prefers Python over Java", "t", 0.82)["memory_id"]

    def test_replace_creates_linked_belief_and_deprecates_old(self):
        new = self.s.replace_memory(self.old, "The user prefers Java", 0.74, reason="contradicted")
        old = self.s.get_memory(self.old)
        self.assertEqual((old["status"], old["superseded_by"], old["version"]), ("deprecated", new["memory_id"], 2))
        self.assertEqual(new["replaces"], [self.old])
        self.assertEqual((new["status"], new["confidence"], new["version"]), ("active", 0.74, 1))
        self.assertEqual(new["source"], f"replaces:{self.old}")

    def test_replace_on_already_deprecated_does_not_bump_version_again(self):
        self.s.update_memory(self.old, status="deprecated")
        self.s.replace_memory(self.old, "The user prefers Java", 0.7)
        self.assertEqual(self.s.get_memory(self.old)["version"], 2)

    def test_near_duplicate_active_belief_is_reused(self):
        existing = self.s.add_memory("The user prefers Java over Python", "t", 0.6)["memory_id"]
        before = len(self.s)
        rep = self.s.replace_memory(self.old, "The user now prefers Java over Python", 0.7)
        self.assertEqual(rep["memory_id"], existing)
        self.assertEqual(len(self.s), before)
        self.assertEqual(self.s.get_memory(existing)["replaces"], [self.old])
        self.assertEqual(self.s.get_memory(existing)["confidence"], 0.6)   # reused as-is

    def test_cannot_be_replaced_twice(self):
        self.s.replace_memory(self.old, "The user prefers Java", 0.7)
        with self.assertRaises(ValueError):
            self.s.replace_memory(self.old, "The user prefers Go", 0.7)

    def test_invalid_input_changes_nothing(self):
        for content, conf in [("  ", 0.5), ("ok", 1.5)]:
            with self.assertRaises(ValueError):
                self.s.replace_memory(self.old, content, conf)
        self.assertEqual(self.s.get_memory(self.old)["status"], "active")
        self.assertEqual(len(self.s), 1)

    def test_unknown_id(self):
        with self.assertRaises(KeyError):
            self.s.replace_memory("mem_9999", "x", 0.5)

    def test_replaced_belief_leaves_default_search_and_replacement_is_found(self):
        self.s.replace_memory(self.old, "The user prefers Java", 0.7)
        found = self.s.find_memories("The user prefers Java")
        self.assertEqual([m["content"] for m in found], ["The user prefers Java"])

    def test_lineage_chain_from_any_member(self):
        b = self.s.replace_memory(self.old, "The user prefers Java", 0.7)["memory_id"]
        c = self.s.replace_memory(b, "The user prefers Rust", 0.7)["memory_id"]
        for start in (self.old, b, c):
            self.assertEqual([m["memory_id"] for m in self.s.get_lineage(start)], [self.old, b, c])

    def test_lineage_of_unlinked_belief_is_itself(self):
        self.assertEqual([m["memory_id"] for m in self.s.get_lineage(self.old)], [self.old])

    def test_links_survive_save_and_load(self):
        new = self.s.replace_memory(self.old, "The user prefers Java", 0.7)["memory_id"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            self.s.save(path)
            s2 = MemoryStore.load(path)
        self.assertEqual(s2.get_memory(self.old)["superseded_by"], new)
        self.assertEqual(s2.get_memory(new)["replaces"], [self.old])


class TestApplyDecisionWithReplacement(unittest.TestCase):
    def setUp(self):
        self.s = MemoryStore()
        self.id = self.s.add_memory("The user prefers Python over Java", "t", 0.82)["memory_id"]

    def dec(self, action="deprecate", rep=None, conf=0.062):
        d = {"memory_id": self.id, "action": action, "new_confidence": conf, "new_content": None, "reason": "r"}
        if rep is not None:
            d["replacement"] = rep
        return d

    def test_deprecate_with_replacement_creates_successor(self):
        old = self.s.apply_decision(self.dec(rep={"content": "The user now prefers Java", "confidence": 0.738}))
        self.assertEqual((old["status"], old["confidence"]), ("deprecated", 0.062))
        new = self.s.get_memory(old["superseded_by"])
        self.assertEqual((new["content"], new["confidence"], new["replaces"]), ("The user now prefers Java", 0.738, [self.id]))

    def test_deprecate_without_replacement_is_unchanged_behaviour(self):
        old = self.s.apply_decision(self.dec())
        self.assertEqual((old["status"], old["superseded_by"], len(self.s)), ("deprecated", None, 1))

    def test_replacement_is_ignored_for_non_deprecate_actions(self):
        self.s.apply_decision(self.dec("update_confidence", rep={"content": "x y", "confidence": 0.5}, conf=0.4))
        self.assertEqual(len(self.s), 1)

    def test_invalid_replacement_leaves_old_belief_untouched(self):
        for bad in [{"content": ""}, {"content": "ok", "confidence": 2}, "oops"]:
            with self.assertRaises(ValueError):
                self.s.apply_decision(self.dec(rep=bad))
        m = self.s.get_memory(self.id)
        self.assertEqual((m["status"], m["version"], len(self.s)), ("active", 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
