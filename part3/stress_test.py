"""
Stress test for Part 3 Revision Engine.

Runs several evidence sequence "patterns" through revise_memory() across a
grid of learning_rate / deprecate_threshold values, to see which parameter
choices produce sensible behavior and which are too twitchy or too sluggish.
"""

from revision_engine import Evidence, revise_memory
from datetime import datetime, timezone

NOW = datetime.now(timezone.utc).isoformat()


def make_ev(relation, strength):
    return Evidence("ev", "mem_001", f"{relation}/{strength}", relation, strength, NOW)


# ---------------------------------------------------------------------
# Evidence patterns to stress-test
# ---------------------------------------------------------------------

PATTERNS = {
    "repeated_strong_contradiction": [make_ev("contradiction", 0.9) for _ in range(5)],
    "repeated_weak_contradiction": [make_ev("contradiction", 0.2) for _ in range(10)],
    "alternating_support_contradiction": [
        make_ev("support", 0.6) if i % 2 == 0 else make_ev("contradiction", 0.6)
        for i in range(8)
    ],
    "repeated_neither": [make_ev("neither", 0.0) for _ in range(5)],
    "single_extreme_contradiction": [make_ev("contradiction", 1.0)],
    "slow_erosion_then_recovery": (
        [make_ev("contradiction", 0.3) for _ in range(4)]
        + [make_ev("support", 0.8) for _ in range(2)]
    ),
}

LEARNING_RATES = [0.2, 0.5, 0.8]
DEPRECATE_THRESHOLDS = [0.10, 0.15, 0.25]

START_CONFIDENCE = 0.82


def run_pattern(evidence_list, learning_rate, deprecate_threshold):
    confidence = START_CONFIDENCE
    trajectory = [confidence]
    deprecated_at = None

    for i, ev in enumerate(evidence_list, start=1):
        decision = revise_memory(
            "mem_001", confidence, ev,
            learning_rate=learning_rate,
            deprecate_threshold=deprecate_threshold,
        )
        if decision.new_confidence is not None:
            confidence = decision.new_confidence
        trajectory.append(confidence)
        if decision.action == "deprecate":
            deprecated_at = i
            break

    return trajectory, deprecated_at


# ---------------------------------------------------------------------
# Run the grid
# ---------------------------------------------------------------------

if __name__ == "__main__":
    for pattern_name, evidence_list in PATTERNS.items():
        print(f"\n=== Pattern: {pattern_name} ({len(evidence_list)} evidence items) ===")
        for lr in LEARNING_RATES:
            for dt in DEPRECATE_THRESHOLDS:
                trajectory, deprecated_at = run_pattern(evidence_list, lr, dt)
                final = trajectory[-1]
                status = f"DEPRECATED at step {deprecated_at}" if deprecated_at else "survived"
                print(
                    f"  lr={lr:<4} threshold={dt:<5} "
                    f"final_confidence={final:<6} {status:<22} "
                    f"trajectory={trajectory}"
                )
