"""
Unit tests for Part 3 — Revision Engine.

Run with: python3 -m unittest test_revision_engine.py -v
"""

import unittest
from datetime import datetime, timezone
from revision_engine import Evidence, revise_memory, RevisionEngine, apply_decision

NOW = datetime.now(timezone.utc).isoformat()


def ev(relation, strength, memory_id="mem_001"):
    return Evidence("ev_test", memory_id, "test evidence", relation, strength, NOW)


class TestReviseMemoryPureFunction(unittest.TestCase):
    """Tests for the stateless revise_memory() update math."""

    def test_neutral_evidence_causes_no_change(self):
        decision = revise_memory("mem_001", 0.7, ev("neutral", 0.9))
        self.assertEqual(decision.action, "no_change")
        self.assertIsNone(decision.new_confidence)

    def test_support_increases_confidence(self):
        decision = revise_memory("mem_001", 0.5, ev("support", 0.6))
        self.assertGreater(decision.new_confidence, 0.5)

    def test_contradiction_decreases_confidence(self):
        decision = revise_memory("mem_001", 0.5, ev("contradiction", 0.6))
        self.assertLess(decision.new_confidence, 0.5)

    def test_confidence_never_exceeds_one(self):
        decision = revise_memory("mem_001", 0.99, ev("support", 1.0), learning_rate=1.0)
        self.assertLessEqual(decision.new_confidence, 1.0)

    def test_confidence_never_goes_below_zero(self):
        decision = revise_memory("mem_001", 0.01, ev("contradiction", 1.0), learning_rate=1.0)
        self.assertGreaterEqual(decision.new_confidence, 0.0)

    def test_zero_strength_evidence_causes_negligible_change(self):
        decision = revise_memory("mem_001", 0.7, ev("contradiction", 0.0))
        self.assertEqual(decision.new_confidence, 0.7)

    def test_stronger_evidence_moves_confidence_further(self):
        weak = revise_memory("mem_001", 0.7, ev("contradiction", 0.2))
        strong = revise_memory("mem_001", 0.7, ev("contradiction", 0.9))
        self.assertLess(strong.new_confidence, weak.new_confidence)

    def test_asymmetric_sensitivity_near_extremes(self):
        # At high confidence, a contradiction should move confidence further
        # than an equal-strength support moves it upward (see point 5 discussion).
        start = 0.82
        support = revise_memory("mem_001", start, ev("support", 0.5))
        contradiction = revise_memory("mem_001", start, ev("contradiction", 0.5))
        support_delta = support.new_confidence - start
        contradiction_delta = start - contradiction.new_confidence
        self.assertGreater(contradiction_delta, support_delta)

    def test_action_is_deprecate_when_below_threshold(self):
        decision = revise_memory(
            "mem_001", 0.2, ev("contradiction", 1.0),
            learning_rate=1.0, deprecate_threshold=0.15,
        )
        self.assertEqual(decision.action, "deprecate")

    def test_action_is_update_confidence_when_above_threshold(self):
        decision = revise_memory(
            "mem_001", 0.5, ev("contradiction", 0.2),
            deprecate_threshold=0.15,
        )
        self.assertEqual(decision.action, "update_confidence")


class TestRevisionEngineSafeguard(unittest.TestCase):
    """Tests for the stateful min_evidence_for_deprecation safeguard."""

    def setUp(self):
        self.engine = RevisionEngine(
            learning_rate=0.8, deprecate_threshold=0.25, min_evidence_for_deprecation=2
        )

    def test_single_strong_contradiction_does_not_deprecate(self):
        decision = self.engine.revise("mem_001", 0.82, ev("contradiction", 1.0))
        self.assertNotEqual(decision.action, "deprecate")
        self.assertEqual(decision.action, "update_confidence")

    def test_second_consecutive_low_confidence_hit_deprecates(self):
        self.engine.revise("mem_001", 0.82, ev("contradiction", 1.0))  # 1st hit: holds
        second_confidence = 0.164  # result of the first hit
        decision = self.engine.revise("mem_001", second_confidence, ev("contradiction", 1.0))
        self.assertEqual(decision.action, "deprecate")

    def test_streak_resets_after_confidence_recovers(self):
        self.engine.revise("mem_001", 0.82, ev("contradiction", 1.0))  # streak = 1
        recovered = self.engine.revise("mem_001", 0.164, ev("support", 1.0))  # confidence jumps back up
        self.assertGreaterEqual(recovered.new_confidence, self.engine.deprecate_threshold)
        # Next contradiction should require a fresh streak, not deprecate immediately.
        decision = self.engine.revise("mem_001", recovered.new_confidence, ev("contradiction", 1.0))
        self.assertNotEqual(decision.action, "deprecate")

    def test_streaks_are_tracked_independently_per_memory(self):
        self.engine.revise("mem_A", 0.82, ev("contradiction", 1.0, memory_id="mem_A"))
        # mem_B should not be affected by mem_A's streak.
        decision_b = self.engine.revise("mem_B", 0.82, ev("contradiction", 1.0, memory_id="mem_B"))
        self.assertNotEqual(decision_b.action, "deprecate")

    def test_min_evidence_of_one_deprecates_immediately(self):
        engine = RevisionEngine(learning_rate=0.8, deprecate_threshold=0.25, min_evidence_for_deprecation=1)
        decision = engine.revise("mem_001", 0.82, ev("contradiction", 1.0))
        self.assertEqual(decision.action, "deprecate")


def part2_evidence(relation, strength, memory_id="mem_0001"):
    """Exactly the dict shape Part 2's collect_evidence() returns."""
    return {
        "evidence_id": "ev_59d9bc7f",
        "target_memory_id": memory_id,
        "new_information": "The user now prefers Java over Python",
        "relation": relation,
        "strength": strength,
        "timestamp": "2026-10-10T20:15:33.113568+00:00",
    }


def memory_obj(confidence=0.82):
    return {
        "memory_id": "mem_0001", "content": "The user prefers Python over Java",
        "confidence": confidence, "created_at": "2026-09-08T10:00:00Z",
        "last_updated": "2026-09-08T10:00:00Z", "source": "conversation_log_12",
        "version": 3, "history": [], "status": "active",
    }


class TestPart2Compatibility(unittest.TestCase):
    """Part 3 must consume Part 2's real output format."""

    def test_accepts_plain_dict_from_part2(self):
        d = revise_memory("mem_0001", 0.82, part2_evidence("contradiction", 0.999))
        self.assertLess(d.new_confidence, 0.82)

    def test_neither_is_treated_as_no_change(self):
        d = revise_memory("mem_0001", 0.82, part2_evidence("neither", 0.0))
        self.assertEqual(d.action, "no_change")

    def test_neutral_alias_still_works(self):
        d = revise_memory("mem_0001", 0.82, part2_evidence("neutral", 0.0))
        self.assertEqual(d.action, "no_change")

    def test_saturated_strength_from_nli_model(self):
        d = revise_memory("mem_0001", 0.82, part2_evidence("support", 1.0), learning_rate=1.0)
        self.assertLessEqual(d.new_confidence, 1.0)

    def test_ignores_extra_fields_relevance_and_gated(self):
        e = part2_evidence("contradiction", 0.9)
        e.update({"relevance": 0.7, "gated": False})
        self.assertLess(revise_memory("mem_0001", 0.82, e).new_confidence, 0.82)

    def test_gated_neither_causes_no_change(self):
        e = part2_evidence("neither", 0.0)
        e.update({"relevance": 0.1, "gated": True})
        self.assertEqual(revise_memory("mem_0001", 0.82, e).action, "no_change")

    def test_rejects_unknown_relation(self):
        with self.assertRaises(ValueError):
            revise_memory("mem_0001", 0.82, part2_evidence("maybe", 0.5))

    def test_rejects_out_of_range_strength(self):
        with self.assertRaises(ValueError):
            revise_memory("mem_0001", 0.82, part2_evidence("support", 1.5))

    def test_rejects_non_numeric_strength(self):
        with self.assertRaises(ValueError):
            revise_memory("mem_0001", 0.82, part2_evidence("support", "high"))

    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            revise_memory("mem_0001", 0.82, {"relation": "support", "strength": 0.5})

    def test_rejects_evidence_for_other_memory(self):
        with self.assertRaises(ValueError):
            revise_memory("mem_0001", 0.82, part2_evidence("support", 0.9, memory_id="mem_0002"))

    def test_decision_serialises_to_shared_schema(self):
        d = revise_memory("mem_0001", 0.82, part2_evidence("contradiction", 0.9)).to_dict()
        self.assertEqual(set(d), {"memory_id", "action", "new_confidence", "new_content", "reason"})


class TestMemoryObjectIntegration(unittest.TestCase):
    """revise_memory_object + apply_decision against the shared Memory Object schema."""

    def setUp(self):
        self.engine = RevisionEngine()

    def test_apply_updates_confidence_version_and_history(self):
        m = memory_obj()
        dec = self.engine.revise_memory_object(m, part2_evidence("contradiction", 0.9))
        new = apply_decision(m, dec)
        self.assertEqual(new["version"], 4)
        self.assertEqual(new["confidence"], dec.new_confidence)
        self.assertEqual(len(new["history"]), 1)
        self.assertEqual(new["history"][0]["confidence"], 0.82)
        self.assertEqual(new["status"], "active")

    def test_apply_does_not_mutate_original(self):
        m = memory_obj()
        dec = self.engine.revise_memory_object(m, part2_evidence("contradiction", 0.9))
        apply_decision(m, dec)
        self.assertEqual(m["version"], 3)
        self.assertEqual(m["history"], [])

    def test_no_change_returns_memory_untouched(self):
        m = memory_obj()
        dec = self.engine.revise_memory_object(m, part2_evidence("neither", 0.0))
        self.assertIs(apply_decision(m, dec), m)

    def test_deprecation_sets_status(self):
        m = memory_obj(0.2)
        eng = RevisionEngine(min_evidence_for_deprecation=1)
        dec = eng.revise_memory_object(m, part2_evidence("contradiction", 0.999))
        self.assertEqual(apply_decision(m, dec)["status"], "deprecated")

    def test_inactive_memory_is_not_revised(self):
        m = memory_obj()
        m["status"] = "deprecated"
        dec = self.engine.revise_memory_object(m, part2_evidence("support", 0.9))
        self.assertEqual(dec.action, "no_change")

    def test_apply_rejects_decision_for_other_memory(self):
        dec = revise_memory("mem_0001", 0.82, part2_evidence("contradiction", 0.9)).to_dict()
        other = memory_obj(); other["memory_id"] = "mem_0002"
        with self.assertRaises(ValueError):
            apply_decision(other, dec)

    def test_full_chain_two_contradictions_deprecate(self):
        m = memory_obj()
        for _ in range(4):
            dec = self.engine.revise_memory_object(m, part2_evidence("contradiction", 0.999))
            m = apply_decision(m, dec)
            if m["status"] == "deprecated":
                break
        self.assertEqual(m["status"], "deprecated")
        self.assertGreaterEqual(m["version"], 5)


class TestReplacementProposal(unittest.TestCase):
    def hit(self, engine, conf, text="The user now prefers Java over Python", strength=0.999):
        e = part2_evidence("contradiction", strength)
        e["new_information"] = text
        return engine.revise("mem_0001", conf, e)

    def test_replacement_proposed_when_belief_is_retired(self):
        eng = RevisionEngine(min_evidence_for_deprecation=2)
        self.hit(eng, 0.2)                       # low hit 1: held
        d = self.hit(eng, 0.1)                   # low hit 2: retired
        self.assertEqual(d.action, "deprecate")
        expected = round(0.5 + 0.5 * 0.999 * 0.5, 3)
        self.assertEqual(d.replacement, {"content": "The user now prefers Java over Python", "confidence": expected})
        self.assertIn("replacement", d.to_dict())

    def test_no_replacement_while_streak_is_still_building(self):
        d = self.hit(RevisionEngine(min_evidence_for_deprecation=2), 0.2)
        self.assertIsNone(d.replacement)
        self.assertNotIn("replacement", d.to_dict())

    def test_replacement_can_be_switched_off(self):
        eng = RevisionEngine(min_evidence_for_deprecation=1, replace_on_deprecate=False)
        d = self.hit(eng, 0.2)
        self.assertEqual(d.action, "deprecate")
        self.assertIsNone(d.replacement)

    def test_no_replacement_without_statement_text(self):
        eng = RevisionEngine(min_evidence_for_deprecation=1)
        d = self.hit(eng, 0.2, text="   ")
        self.assertEqual(d.action, "deprecate")
        self.assertIsNone(d.replacement)

    def test_stronger_evidence_gives_higher_starting_confidence(self):
        weak = self.hit(RevisionEngine(min_evidence_for_deprecation=1), 0.1, strength=0.3).replacement
        strong = self.hit(RevisionEngine(min_evidence_for_deprecation=1), 0.1, strength=0.99).replacement
        self.assertLess(weak["confidence"], strong["confidence"])

    def test_replacement_prior_is_configurable(self):
        eng = RevisionEngine(min_evidence_for_deprecation=1, replacement_prior=0.3)
        d = self.hit(eng, 0.2)
        self.assertEqual(d.replacement["confidence"], round(0.3 + 0.5 * 0.999 * 0.7, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
