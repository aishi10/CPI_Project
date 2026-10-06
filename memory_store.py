from datetime import datetime
import re


# ============================================================
# MEMORY LAYER
# ============================================================

memory = []
revision_history = []


def add_memory(subject, fact, confidence, source):
    memory.append({
        "subject": subject,
        "fact": fact,
        "confidence": confidence,
        "source": source,
        "status": "active"
    })


def get_memories(subject):
    return [
        m for m in memory
        if m["subject"].lower() == subject.lower()
    ]


# ============================================================
# EVIDENCE LAYER
# ============================================================

def add_evidence(
    subject,
    fact,
    strength,
    source,
    reliability,
    evidence_date,
    valid_until=None,
    source_group=None
):
    return {
        "subject": subject,
        "fact": fact,
        "strength": strength,
        "source": source,
        "reliability": reliability,
        "evidence_date": evidence_date,
        "valid_until": valid_until,
        "source_group": source_group
    }


# ============================================================
# SOURCE INDEPENDENCE
# ============================================================

def calculate_source_independence(evidence_list):

    source_groups = set()

    for evidence in evidence_list:

        if evidence["source_group"] is not None:
            source_groups.add(
                evidence["source_group"]
            )

    if len(source_groups) == 0:
        return 0.0

    return min(
        len(source_groups) / 3,
        1.0
    )


# ============================================================
# RECENCY
# ============================================================

def calculate_recency(evidence_date):

    today = datetime.now().date()

    evidence_day = datetime.strptime(
        evidence_date,
        "%Y-%m-%d"
    ).date()

    age_days = (
        today - evidence_day
    ).days

    if age_days <= 7:
        return 1.0

    elif age_days <= 30:
        return 0.9

    elif age_days <= 90:
        return 0.7

    elif age_days <= 365:
        return 0.5

    else:
        return 0.2


# ============================================================
# TEMPORAL VALIDITY
# ============================================================

def calculate_temporal_validity(valid_until):

    if valid_until is None:
        return 1.0

    today = datetime.now().date()

    expiry_date = datetime.strptime(
        valid_until,
        "%Y-%m-%d"
    ).date()

    if today <= expiry_date:
        return 1.0

    return 0.0


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    filler_words = [
        "the",
        "a",
        "an",
        "now",
        "currently",
        "is",
        "are",
        "was",
        "were"
    ]

    words = text.split()

    words = [
        word
        for word in words
        if word not in filler_words
    ]

    return " ".join(words)


# ============================================================
# TIME EXTRACTION
# ============================================================

def extract_time(text):

    pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)'

    match = re.search(
        pattern,
        text.lower()
    )

    if match:

        hour = int(match.group(1))

        minute = (
            int(match.group(2))
            if match.group(2)
            else 0
        )

        period = match.group(3)

        return hour, minute, period

    return None


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_number(text):

    pattern = r'(-?\d+(?:\.\d+)?)'

    match = re.search(
        pattern,
        text
    )

    if match:
        return float(match.group(1))

    return None


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_contradiction(memory, evidence):

    memory_fact = memory["fact"].lower()
    evidence_fact = evidence["fact"].lower()

    # --------------------------------------------------------
    # 1. TIME
    # --------------------------------------------------------

    memory_time = extract_time(memory_fact)
    evidence_time = extract_time(evidence_fact)

    if memory_time and evidence_time:

        if memory_time != evidence_time:
            return True

        return False

    # --------------------------------------------------------
    # 2. OPEN / CLOSED
    # --------------------------------------------------------

    open_words = [
        "open",
        "opened",
        "available",
        "accessible"
    ]

    closed_words = [
        "closed",
        "shut",
        "unavailable",
        "inaccessible"
    ]

    memory_open = any(
        word in memory_fact
        for word in open_words
    )

    memory_closed = any(
        word in memory_fact
        for word in closed_words
    )

    evidence_open = any(
        word in evidence_fact
        for word in open_words
    )

    evidence_closed = any(
        word in evidence_fact
        for word in closed_words
    )

    if memory_open and evidence_closed:
        return True

    if memory_closed and evidence_open:
        return True

    # --------------------------------------------------------
    # 3. YES / NO
    # --------------------------------------------------------

    positive_words = [
        "yes",
        "true",
        "allowed",
        "enabled",
        "supports"
    ]

    negative_words = [
        "no",
        "false",
        "not allowed",
        "disabled",
        "does not support"
    ]

    memory_positive = any(
        word in memory_fact
        for word in positive_words
    )

    memory_negative = any(
        word in memory_fact
        for word in negative_words
    )

    evidence_positive = any(
        word in evidence_fact
        for word in positive_words
    )

    evidence_negative = any(
        word in evidence_fact
        for word in negative_words
    )

    if memory_positive and evidence_negative:
        return True

    if memory_negative and evidence_positive:
        return True

    # --------------------------------------------------------
    # 4. NUMERIC VALUES
    # --------------------------------------------------------

    memory_number = extract_number(memory_fact)
    evidence_number = extract_number(evidence_fact)

    if (
        memory_number is not None
        and evidence_number is not None
    ):

        if memory_number != evidence_number:

            memory_words = set(
                normalize_text(memory_fact).split()
            )

            evidence_words = set(
                normalize_text(evidence_fact).split()
            )

            common_words = (
                memory_words.intersection(
                    evidence_words
                )
            )

            if len(common_words) >= 2:
                return True

    # --------------------------------------------------------
    # 5. LOCATION
    # --------------------------------------------------------

    location_patterns = [
        r'lives in ([a-z ]+)',
        r'located in ([a-z ]+)',
        r'based in ([a-z ]+)',
        r'from ([a-z ]+)'
    ]

    for pattern in location_patterns:

        memory_match = re.search(
            pattern,
            memory_fact
        )

        evidence_match = re.search(
            pattern,
            evidence_fact
        )

        if memory_match and evidence_match:

            memory_location = (
                memory_match.group(1).strip()
            )

            evidence_location = (
                evidence_match.group(1).strip()
            )

            if memory_location != evidence_location:
                return True

    return False


# ============================================================
# SINGLE EVIDENCE EVALUATION
# ============================================================

def evaluate_evidence(memory, evidence):

    contradicts = detect_contradiction(
        memory,
        evidence
    )

    recency = calculate_recency(
        evidence["evidence_date"]
    )

    temporal_validity = calculate_temporal_validity(
        evidence["valid_until"]
    )

    evidence_score = (
        0.4 * evidence["strength"]
        + 0.3 * evidence["reliability"]
        + 0.2 * recency
        + 0.1 * temporal_validity
    )

    return {
        "evidence_score": evidence_score,
        "recency": recency,
        "temporal_validity": temporal_validity,
        "contradicts": contradicts
    }


# ============================================================
# GROUP EVIDENCE BY FACT
# ============================================================

def group_evidence_by_fact(evidence_list):

    groups = {}

    for evidence in evidence_list:

        fact = normalize_text(
            evidence["fact"]
        )

        if fact not in groups:
            groups[fact] = []

        groups[fact].append(evidence)

    return groups


# ============================================================
# SCORE EACH COMPETING BELIEF
# ============================================================

def score_evidence_group(
    memory,
    evidence_group
):

    total_score = 0

    source_groups = set()

    evaluations = []

    for evidence in evidence_group:

        evaluation = evaluate_evidence(
            memory,
            evidence
        )

        evaluations.append(
            evaluation
        )

        total_score += (
            evaluation["evidence_score"]
        )

        if evidence["source_group"] is not None:
            source_groups.add(
                evidence["source_group"]
            )

    # Average strength of evidence supporting
    # this particular fact
    average_score = (
        total_score / len(evidence_group)
    )

    # Number of independent source groups
    independent_sources = len(
        source_groups
    )

    # Independence bonus
    independence_score = min(
        independent_sources / 3,
        1.0
    )

    # Final support score
    support_score = (
        0.7 * average_score
        + 0.3 * independence_score
    )

    return {
        "evidence": evidence_group,
        "evaluations": evaluations,
        "average_score": average_score,
        "independent_sources": independent_sources,
        "independence_score": independence_score,
        "support_score": support_score
    }


# ============================================================
# COMPARE COMPETING BELIEFS
# ============================================================

def evaluate_competing_evidence(
    memory,
    evidence_list
):

    groups = group_evidence_by_fact(
        evidence_list
    )

    scored_groups = []

    for fact, evidence_group in groups.items():

        result = score_evidence_group(
            memory,
            evidence_group
        )

        result["fact"] = fact

        scored_groups.append(
            result
        )

    # Sort strongest belief first
    scored_groups.sort(
        key=lambda x: x["support_score"],
        reverse=True
    )

    return scored_groups


# ============================================================
# REVISION DECISION
# ============================================================

def decide_multiple_evidence_revision(
    memory,
    evidence_list
):

    scored_groups = evaluate_competing_evidence(
        memory,
        evidence_list
    )

    if len(scored_groups) == 0:
        return {
            "decision": "discard",
            "winning_fact": None,
            "support_score": 0
        }

    winner = scored_groups[0]

    # Strong winning belief
    if winner["support_score"] >= 0.85:

        return {
            "decision": "revise",
            "winning_fact": winner["fact"],
            "support_score": winner["support_score"],
            "winning_group": winner,
            "all_groups": scored_groups
        }

    # Medium confidence
    elif winner["support_score"] >= 0.60:

        return {
            "decision": "remain_uncertain",
            "winning_fact": winner["fact"],
            "support_score": winner["support_score"],
            "winning_group": winner,
            "all_groups": scored_groups
        }

    # Weak evidence
    else:

        return {
            "decision": "discard",
            "winning_fact": winner["fact"],
            "support_score": winner["support_score"],
            "winning_group": winner,
            "all_groups": scored_groups
        }


# ============================================================
# ACTUAL MEMORY REVISION
# ============================================================

def revise_memory(memory, evidence):

    old_fact = memory["fact"]

    memory["fact"] = evidence["fact"]

    memory["confidence"] = evidence["strength"]

    memory["source"] = evidence["source"]

    return old_fact, memory


# ============================================================
# REVISION HISTORY
# ============================================================

def record_revision(
    old_memory,
    evidence,
    evaluation,
    decision
):

    revision_history.append({

        "old_fact": old_memory["fact"],

        "new_fact": evidence["fact"],

        "old_confidence": (
            old_memory["confidence"]
        ),

        "new_confidence": evidence["strength"],

        "old_source": (
            old_memory["source"]
        ),

        "new_source": evidence["source"],

        "evidence_score": (
            evaluation["evidence_score"]
        ),

        "decision": decision
    })


# ============================================================
# TEST
# ============================================================

add_memory(
    "library",
    "library closes at 6 PM",
    0.80,
    "Old library notice"
)


# ------------------------------------------------------------
# Evidence 1
# ------------------------------------------------------------

evidence1 = add_evidence(
    "library",
    "library closes at 8 PM",
    0.95,
    "Official library website",
    0.95,
    "2026-10-04",
    "2026-10-31",
    "library_authority"
)


# ------------------------------------------------------------
# Evidence 2
# ------------------------------------------------------------

evidence2 = add_evidence(
    "library",
    "library closes at 8 PM",
    0.90,
    "Official library notice",
    0.90,
    "2026-10-04",
    "2026-10-31",
    "library_authority"
)


# ------------------------------------------------------------
# Evidence 3
# ------------------------------------------------------------

evidence3 = add_evidence(
    "library",
    "library closes at 10 PM",
    0.50,
    "Random online comment",
    0.30,
    "2026-10-04",
    "2026-10-31",
    "random_user"
)


evidence_list = [
    evidence1,
    evidence2,
    evidence3
]


# ============================================================
# RUN COMPETING EVIDENCE EVALUATION
# ============================================================

result = decide_multiple_evidence_revision(
    memory[0],
    evidence_list
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nCompeting Evidence Evaluation")
print("--------------------------------")

print(
    "Decision:",
    result["decision"]
)

print(
    "Winning fact:",
    result["winning_fact"]
)

print(
    "Winning support score:",
    result["support_score"]
)


print("\nCompeting Beliefs")
print("--------------------------------")


for i, group in enumerate(
    result["all_groups"],
    start=1
):

    print(
        f"\nBelief {i}:"
    )

    print(
        "Fact:",
        group["evidence"][0]["fact"]
    )

    print(
        "Number of evidence items:",
        len(group["evidence"])
    )

    print(
        "Independent source groups:",
        group["independent_sources"]
    )

    print(
        "Average evidence score:",
        group["average_score"]
    )

    print(
        "Independence score:",
        group["independence_score"]
    )

    print(
        "Support score:",
        group["support_score"]
    )
