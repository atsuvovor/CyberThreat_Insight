def executive_summary_prompt(facts: list[str], actions: list[str]) -> str:
    """
    Builds a structured executive summary prompt.
    """

    facts_block = "\n".join(f"- {fact}" for fact in facts[:4])
    actions_block = "\n".join(f"- {action}" for action in actions[:3])

    return f"""
Write a professional cybersecurity executive summary using ONLY the facts below.

Do NOT:
- Repeat instructions
- Ask questions
- Introduce new information

FORMAT (do not change headers):

Summary:
<2–3 concise sentences>

Data Analysis Key Findings:
{facts_block}

Insights or Next Steps:
{actions_block}

Additional Observations from Charts:
- {facts[-1]}
"""
