
def build_prompt(memory_content, new_information):
    """
    Build a fixed prompt for classifying incoming evidence.
    The LLM only classifies evidence; it does not revise memory.
    """

    prompt = f"""
You are an evidence classifier in a memory revision system.

Compare the EXISTING MEMORY with the NEW INFORMATION.

EXISTING MEMORY:
{memory_content}

NEW INFORMATION:
{new_information}

Classify the relationship between the new information and the memory.

Choose exactly one relation:
- support: The new information provides evidence in favor of the memory.
- contradiction: The new information provides evidence against the memory.
- neither: The new information does not meaningfully support or contradict the memory.

Important rules:
1. Learning something does not necessarily mean preferring it.
2. A change in preference may contradict an earlier preference.
3. New information about an unrelated topic is neither.
4. Do not assume facts that are not provided.
5. Do not update, rewrite, or delete the memory.
6. Do not recommend a memory revision action.

Assign a strength between 0.0 and 1.0:
- 0.0: No meaningful evidence.
- 0.1-0.3: Weak evidence.
- 0.4-0.6: Moderate evidence.
- 0.7-0.85: Strong evidence.
- 0.86-1.0: Very strong evidence.

For relation "neither", strength must be 0.0.
For support or contradiction, strength indicates how strongly
the new information bears on the truth of the existing memory.

Return ONLY a valid JSON object in this exact format:
{{
    "relation": "support",
    "strength": 0.8
}}
"""
    return prompt

