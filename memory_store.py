"""
Part 1 - Memory Store & Data Model
Adaptive Memory Evolution through Evidence-Based Revision

Holds every belief as a Memory Object (shared schema), keeps a full version
history, and exposes the operations Parts 2-4 need.

Memory Object:
    memory_id, content, confidence (0-1), created_at, last_updated,
    source, version, history[], status (active | deprecated | merged),
    replaces[] (ids of beliefs this one replaced), superseded_by (id or None)
    replaces / superseded_by are link metadata: changing them does not bump version.

History entry (matches Part 3's apply_decision):
    {"version", "confidence", "content", "updated_at", "reason"}
    "reason" explains why that version was REPLACED (the update that followed it).
"""

import copy
import json
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple, Union

from similarity import jaccard_similarity

VALID_STATUSES = {"active", "deprecated", "merged"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be a number in [0, 1], got {value!r}")
    return float(value)


class MemoryStore:
    def __init__(self, similarity: Optional[Callable[[str, str], float]] = None,
                 dedupe_threshold: float = 0.8):
        """
        similarity(a, b) -> 0..1 ranks memories in find_memories (default: word-overlap Jaccard).
        dedupe_threshold: replace_memory reuses an active belief at least this similar instead of
        creating a near-duplicate.
        """
        self._dedupe = dedupe_threshold
        self._memories: Dict[str, dict] = {}
        self._counter = 0
        self._sim = similarity or jaccard_similarity

    # ------------------------------------------------------------------
    # Create / read
    # ------------------------------------------------------------------

    def add_memory(self, content: str, source: str, confidence: float = 0.5) -> dict:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        confidence = _check_confidence(confidence)

        self._counter += 1
        now = _now()
        memory = {
            "memory_id": f"mem_{self._counter:04d}",
            "content": content.strip(),
            "confidence": confidence,
            "created_at": now,
            "last_updated": now,
            "source": source,
            "version": 1,
            "history": [],
            "status": "active",
            "replaces": [],
            "superseded_by": None,
        }
        self._memories[memory["memory_id"]] = memory
        return copy.deepcopy(memory)

    def get_memory(self, memory_id: str) -> Optional[dict]:
        m = self._memories.get(memory_id)
        return copy.deepcopy(m) if m else None

    def list_memories(self, status: Optional[str] = "active") -> List[dict]:
        """All memories with the given status (None = every status), oldest first."""
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return [copy.deepcopy(m) for m in self._memories.values()
                if status is None or m["status"] == status]

    def get_history(self, memory_id: str) -> List[dict]:
        """Past versions plus the current one, oldest first."""
        m = self._require(memory_id)
        current = {"version": m["version"], "confidence": m["confidence"],
                   "content": m["content"], "updated_at": m["last_updated"],
                   "reason": "current"}
        return copy.deepcopy(m["history"]) + [current]

    def find_memories_scored(self, query: str, limit: int = 5,
                             status: Optional[str] = "active") -> List[Tuple[dict, float]]:
        """(memory, similarity) pairs, best first. Zero-similarity memories are left out."""
        scored = []
        for m in self.list_memories(status):
            score = self._sim(query, m["content"])
            if score > 0:
                scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def find_memories(self, query: str, limit: int = 5, status: Optional[str] = "active") -> List[dict]:
        """Memories most similar to `query`; use to pick the target of new information."""
        return [m for m, _ in self.find_memories_scored(query, limit, status)]

    # ------------------------------------------------------------------
    # Update (always versioned)
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, new_content: Optional[str] = None,
                      new_confidence: Optional[float] = None, status: Optional[str] = None,
                      reason: str = "") -> dict:
        """
        Apply a change. The previous state is pushed onto history and the version
        increments. Passing nothing to change raises, so versions never bump for no reason.
        """
        m = self._require(memory_id)

        if new_content is None and new_confidence is None and status is None:
            raise ValueError("nothing to update")
        if new_content is not None and (not isinstance(new_content, str) or not new_content.strip()):
            raise ValueError("new_content must be a non-empty string")
        if new_confidence is not None:
            new_confidence = _check_confidence(new_confidence)
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")

        m["history"].append({
            "version": m["version"],
            "confidence": m["confidence"],
            "content": m["content"],
            "updated_at": m["last_updated"],
            "reason": reason or "update",
        })
        if new_content is not None:
            m["content"] = new_content.strip()
        if new_confidence is not None:
            m["confidence"] = new_confidence
        if status is not None:
            m["status"] = status
        m["version"] += 1
        m["last_updated"] = _now()
        return copy.deepcopy(m)

    def apply_decision(self, decision: dict) -> dict:
        """
        Execute a Revision Decision from Part 3:
            {memory_id, action, new_confidence, new_content, reason}
        update_confidence / overwrite -> update; deprecate -> update + status deprecated;
        no_change -> returns the memory untouched (no version bump).
        """
        required = {"memory_id", "action"}
        if not required <= set(decision):
            raise ValueError(f"decision missing fields: {sorted(required - set(decision))}")

        mid, action = decision["memory_id"], decision["action"]
        reason = decision.get("reason", "")
        m = self._require(mid)

        if action == "no_change":
            return copy.deepcopy(m)
        if action in ("update_confidence", "overwrite"):
            return self.update_memory(mid, decision.get("new_content"),
                                      decision.get("new_confidence"), reason=reason)
        if action == "deprecate":
            rep = decision.get("replacement")
            if rep is not None:   # validate first so a bad replacement cannot leave a half-applied change
                if not isinstance(rep, dict) or not isinstance(rep.get("content"), str) or not rep["content"].strip():
                    raise ValueError("replacement needs a non-empty 'content'")
                _check_confidence(rep.get("confidence", 0.5))
                if m.get("superseded_by"):
                    raise ValueError(f"{mid} is already superseded by {m['superseded_by']}")
            self.update_memory(mid, decision.get("new_content"),
                               decision.get("new_confidence"), status="deprecated", reason=reason)
            if rep is not None:
                self.replace_memory(mid, rep["content"], rep.get("confidence", 0.5), reason=reason)
            return copy.deepcopy(self._memories[mid])   # carries superseded_by when a replacement exists
        raise ValueError(f"unknown action {action!r}")

    def replace_memory(self, old_id: str, new_content: str, confidence: float = 0.5,
                       reason: str = "") -> dict:
        """
        Retire `old_id` (if still active) and link it to the belief that takes its place.
        If an active belief is already at least `dedupe_threshold` similar to `new_content`, that
        belief is reused instead of creating a near-duplicate. Returns the replacement.
        """
        old = self._require(old_id)
        if old.get("superseded_by"):
            raise ValueError(f"{old_id} is already superseded by {old['superseded_by']}")
        if not isinstance(new_content, str) or not new_content.strip():
            raise ValueError("new_content must be a non-empty string")
        confidence = _check_confidence(confidence)

        if old["status"] == "active":
            self.update_memory(old_id, status="deprecated", reason=reason or "replaced")

        dup = None
        best = 0.0
        for m in self._memories.values():
            if m["status"] != "active" or m["memory_id"] == old_id:
                continue
            score = self._sim(new_content, m["content"])
            if score >= self._dedupe and score > best:
                dup, best = m, score

        if dup is not None:
            target = dup
        else:
            created = self.add_memory(new_content, source=f"replaces:{old_id}", confidence=confidence)
            target = self._memories[created["memory_id"]]

        old["superseded_by"] = target["memory_id"]
        target.setdefault("replaces", [])
        if old_id not in target["replaces"]:
            target["replaces"].append(old_id)
        return copy.deepcopy(target)

    def get_lineage(self, memory_id: str) -> List[dict]:
        """Chain of beliefs around `memory_id`, oldest first: what it replaced, itself, what replaced it."""
        m = self._require(memory_id)
        seen = {memory_id}
        back, cur = [], m
        while cur.get("replaces"):
            prev = cur["replaces"][0]
            if prev in seen or prev not in self._memories:
                break
            seen.add(prev)
            cur = self._memories[prev]
            back.append(cur)
        fwd, cur = [], m
        while cur.get("superseded_by"):
            nxt = cur["superseded_by"]
            if nxt in seen or nxt not in self._memories:
                break
            seen.add(nxt)
            cur = self._memories[nxt]
            fwd.append(cur)
        return [copy.deepcopy(x) for x in list(reversed(back)) + [m] + fwd]

    def mark_merged(self, memory_id: str, into_id: str) -> dict:
        """Retire a duplicate memory by pointing it at the one that absorbed it."""
        self._require(into_id)
        return self.update_memory(memory_id, status="merged", reason=f"merged into {into_id}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"counter": self._counter, "memories": self._memories}, f, indent=2)

    @classmethod
    def load(cls, path: str, similarity=None, dedupe_threshold: float = 0.8) -> "MemoryStore":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        store = cls(similarity, dedupe_threshold)
        store._counter = data["counter"]
        store._memories = data["memories"]
        return store

    # ------------------------------------------------------------------

    def _require(self, memory_id: str) -> dict:
        if memory_id not in self._memories:
            raise KeyError(f"unknown memory_id {memory_id!r}")
        return self._memories[memory_id]

    def __len__(self) -> int:
        return len(self._memories)


# Module-level convenience API (matches the signatures in the original contract)
_default = MemoryStore()

def add_memory(content: str, source: str, confidence: float = 0.5) -> dict:
    return _default.add_memory(content, source, confidence)

def get_memory(memory_id: str) -> Optional[dict]:
    return _default.get_memory(memory_id)

def update_memory(memory_id: str, new_content: Optional[str] = None,
                  new_confidence: Optional[float] = None, **kw) -> dict:
    return _default.update_memory(memory_id, new_content, new_confidence, **kw)
