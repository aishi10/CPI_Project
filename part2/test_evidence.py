
from evidence_collector import collect_evidence

memory = {
    "memory_id": "mem_0001",
    "content": "The user prefers Python over Java",
    "confidence": 0.82,
    "created_at": "2026-09-08T10:00:00Z",
    "last_updated": "2026-09-08T10:00:00Z",
    "source": "conversation_log_12",
    "version": 3,
    "history": [],
    "status": "active"
}

new_information = "User just said they're learning Java for a job"

evidence = collect_evidence(memory, new_information)

print("Generated Evidence Object:")
print(evidence)

