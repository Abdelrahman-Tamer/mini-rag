from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template("\n".join([
    "You are a travel assistant. Your knowledge comes ONLY from the provided sources below.",
    "",
    "RULES:",
    "1. Answer ONLY using information EXPLICITLY stated in the provided sources.",
    "2. If the sources only partially answer, state what is confirmed and note what is missing. Don't fabricate facts or make up information not present in the sources.",
    "3. Respond in the same language as the user.",
    "4. Keep answers clear, direct, and concise.",
]))

#### Document ####
document_prompt = Template(
    "\n".join([
        "## Source $doc_num [Relevance: $score]",
        "$chunk_text",
    ])
)

#### Footer ####
footer_prompt = Template("\n".join([
    "REMEMBER: Answer ONLY from the provided sources above. If they do not answer the question, say so.",
    "## Question:",
    "$query",
    "",
    "## Answer:",
]))

related_prompt = Template("\n".join([
    "Based ONLY on the provided sources, suggest 3-5 follow-up questions the user might ask.",
    "Only suggest questions the sources can actually answer. Return a simple numbered list — nothing else.",
    "## Question:",
    "$query",
    "## Answer:",
    "$answer",
    "",
    "## Suggested Questions:",
]))
