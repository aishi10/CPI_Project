
from datetime import datetime, timezone
import uuid


def collect_evidence(memory, new_information):
    """
    Compare new information with an existing memory
    and return an Evidence Object.

    Currently uses mock LLM output.
    We will connect the pretrained LLM later.
    """

    # TEMPORARY: Mock LLM result for testing
    relation = "contradiction"
    strength = 0.7

    # Construct the Evidence Object using the shared schema
    evidence = {
        "evidence_id": f"ev_{uuid.uuid4().hex[:8]}",
        "target_memory_id": memory["memory_id"],
        "new_information": new_information,
        "relation": relation,
        "strength": strength,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return evidence


