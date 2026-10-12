"""
Part 3 - Revision Engine
Adaptive Memory Evolution through Evidence-Based Revision

Input : an Evidence Object (dict from Part 2's collect_evidence) and a Memory
        Object (dict, shared schema from Part 1).
Output: a Revision Decision (dict, shared schema) and, optionally, the updated
        Memory Object with version history.

Part 2 emits relation in {"support", "contradiction", "neither"}; the original
schema said "neutral". Both are accepted and treated identically.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Union

RELATION_ALIASES = {
    "support": "support",
    "contradiction": "contradiction",
    "neither": "neutral",   # Part 2 vocabulary
    "neutral": "neutral",   # original shared-schema vocabulary
}


# ---------------------------------------------------------------------
# Schema objects
# ---------------------------------------------------------------------

@dataclass
class Evidence:
    evidence_id: str
    target_memory_id: str
    new_information: str
    relation: str
    strength: float
    timestamp: str

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        missing = [k for k in ("evidence_id", "target_memory_id", "relation", "strength") if k not in d]
        if missing:
            raise ValueError(f"Evidence Object missing fields: {missing}")
        return cls(
            evidence_id=d["evidence_id"],
            target_memory_id=d["target_memory_id"],
            new_information=d.get("new_information", ""),
            relation=d["relation"],
            strength=d["strength"],
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class RevisionDecision:
    memory_id: str
    action: str             # update_confidence | deprecate | no_change  (overwrite is unused)
    new_confidence: Optional[float]
    new_content: Optional[str]
    reason: str
    # {"content": str, "confidence": float}; only set when a belief is retired because of a
    # contradiction. The Memory Store (Part 1) turns it into the replacement belief.
    replacement: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["replacement"] is None:
            del d["replacement"]     # keeps the original 5-field schema when there is no replacement
        return d


def _coerce_evidence(evidence: Union[Evidence, dict]) -> Evidence:
    ev = evidence if isinstance(evidence, Evidence) else Evidence.from_dict(evidence)

    if ev.relation not in RELATION_ALIASES:
        raise ValueError(f"Unknown relation {ev.relation!r}; expected one of {sorted(RELATION_ALIASES)}")
    s = ev.strength
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0.0 <= s <= 1.0:
        raise ValueError(f"strength must be a number in [0, 1], got {s!r}")

    return Evidence(ev.evidence_id, ev.target_memory_id, ev.new_information,
                    RELATION_ALIASES[ev.relation], float(s), ev.timestamp)


# ---------------------------------------------------------------------
# Core update rule (stateless)
# ---------------------------------------------------------------------

def revise_memory(
    memory_id: str,
    current_confidence: float,
    evidence: Union[Evidence, dict],
    learning_rate: float = 0.5,
    deprecate_threshold: float = 0.15,
) -> RevisionDecision:
    """
    target = 1.0 (support) / 0.0 (contradiction); neutral -> no change.
    new_confidence = c + learning_rate * strength * (target - c), clamped to [0, 1].
    """
    ev = _coerce_evidence(evidence)

    if ev.target_memory_id != memory_id:
        raise ValueError(f"Evidence targets {ev.target_memory_id!r} but memory is {memory_id!r}")

    if ev.relation == "neutral":
        return RevisionDecision(memory_id, "no_change", None, None,
                                "Evidence is neutral ('neither'); no update applied.")

    target = 1.0 if ev.relation == "support" else 0.0
    delta = learning_rate * ev.strength * (target - current_confidence)
    new_conf = round(max(0.0, min(1.0, current_confidence + delta)), 3)

    if new_conf < deprecate_threshold:
        return RevisionDecision(
            memory_id, "deprecate", new_conf, None,
            f"Confidence fell to {new_conf} after {ev.relation} evidence (strength={ev.strength}); "
            f"below deprecate threshold ({deprecate_threshold}).")

    return RevisionDecision(
        memory_id, "update_confidence", new_conf, None,
        f"{ev.relation.capitalize()} evidence (strength={ev.strength}) shifted confidence "
        f"from {current_confidence} to {new_conf}.")


# ---------------------------------------------------------------------
# Stateful engine: requires N consecutive low-confidence hits to deprecate
# ---------------------------------------------------------------------

class RevisionEngine:
    def __init__(self, learning_rate: float = 0.5, deprecate_threshold: float = 0.15,
                 min_evidence_for_deprecation: int = 2,
                 replace_on_deprecate: bool = True, replacement_prior: float = 0.5):
        """
        replace_on_deprecate: when a belief is retired, propose a replacement built from the
        contradicting statement. Its starting confidence is `replacement_prior` nudged toward 1 by
        that same evidence (the evidence against the old belief is evidence for the new one).
        """
        self.learning_rate = learning_rate
        self.deprecate_threshold = deprecate_threshold
        self.min_evidence_for_deprecation = min_evidence_for_deprecation
        self.replace_on_deprecate = replace_on_deprecate
        self.replacement_prior = replacement_prior
        self._low_streak: dict = {}

    def revise(self, memory_id: str, current_confidence: float,
               evidence: Union[Evidence, dict]) -> RevisionDecision:
        decision = revise_memory(memory_id, current_confidence, evidence,
                                 self.learning_rate, self.deprecate_threshold)

        if decision.action != "deprecate":
            if decision.new_confidence is not None and decision.new_confidence >= self.deprecate_threshold:
                self._low_streak[memory_id] = 0
            return decision

        streak = self._low_streak.get(memory_id, 0) + 1
        self._low_streak[memory_id] = streak

        if streak >= self.min_evidence_for_deprecation:
            decision.reason += (f" Streak of {streak} low-confidence hits reached minimum "
                                f"({self.min_evidence_for_deprecation}) - deprecating.")
            if self.replace_on_deprecate:
                ev = _coerce_evidence(evidence)
                text = (ev.new_information or "").strip()
                if ev.relation == "contradiction" and text:
                    conf = round(min(1.0, self.replacement_prior
                                     + self.learning_rate * ev.strength * (1 - self.replacement_prior)), 3)
                    decision.replacement = {"content": text, "confidence": conf}
                    decision.reason += f" Proposing a replacement belief at confidence {conf}."
            return decision

        return RevisionDecision(
            memory_id, "update_confidence", decision.new_confidence, None,
            decision.reason + f" Below threshold but streak ({streak}/{self.min_evidence_for_deprecation}) "
                              "not yet reached - holding instead of deprecating.")

    # --- Memory Object level (shared schema) --------------------------

    def revise_memory_object(self, memory: dict, evidence: Union[Evidence, dict]) -> RevisionDecision:
        """Decide a revision for a Memory Object dict. Inactive memories are left alone."""
        if memory.get("status", "active") != "active":
            return RevisionDecision(memory["memory_id"], "no_change", None, None,
                                    f"Memory status is {memory['status']!r}; not revising.")
        return self.revise(memory["memory_id"], memory["confidence"], evidence)


def apply_decision(memory: dict, decision: Union[RevisionDecision, dict]) -> dict:
    """
    Return a NEW Memory Object with the decision applied: confidence/content updated,
    version incremented, previous state pushed onto history, status set on deprecate.
    no_change decisions return the memory untouched.
    """
    d = decision.to_dict() if isinstance(decision, RevisionDecision) else decision
    if d["memory_id"] != memory["memory_id"]:
        raise ValueError("Decision targets a different memory")
    if d["action"] == "no_change":
        return memory

    now = datetime.now(timezone.utc).isoformat()
    updated = dict(memory)
    updated["history"] = list(memory.get("history", [])) + [{
        "version": memory.get("version", 1),
        "confidence": memory["confidence"],
        "content": memory["content"],
        "updated_at": memory.get("last_updated", now),
    }]
    updated["version"] = memory.get("version", 1) + 1
    updated["last_updated"] = now

    if d.get("new_confidence") is not None:
        updated["confidence"] = d["new_confidence"]
    if d.get("new_content") is not None:
        updated["content"] = d["new_content"]
    if d["action"] == "deprecate":
        updated["status"] = "deprecated"
    return updated
