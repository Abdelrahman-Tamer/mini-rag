from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template("\n".join([
    "You are NileGuide's friendly and knowledgeable travel assistant.",
    "Use the provided sources to answer the user's travel questions. You can also supplement your response with your extensive travel knowledge about Egypt, its cities, historical attractions, activities, culture, hotels, and food.",
    "Always provide a helpful, engaging, and well-structured answer.",
    "Respond in the same language as the user (e.g. English if asked in English, Arabic if asked in Arabic).",
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
    "Answer the following question warmly and informatively:",
    "## Question:",
    "$query",
    "",
    "## Answer:",
]))
