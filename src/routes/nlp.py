from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import ChatRequest
from controllers import NLPController
from models import ResponseSignal
import os

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


@nlp_router.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    prior_history = chat_request.history or []

    # validate non-empty query
    text = chat_request.text.strip() if chat_request.text else ""
    if not text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.INVALID_QUERY.value,
                "message": "Please enter a valid question.",
            }
        )

    # handle greetings: skip retrieval, respond naturally
    if nlp_controller.is_greeting_query(chat_request.text):
        greeting_system = (
            "You are a friendly travel assistant. "
            "The user just greeted you. Respond warmly in the same language as them. "
            "Offer to help with travel questions about cities, attractions, food, hotels, and shopping. "
            "Keep it short and welcoming."
        )
        answer, full_prompt, _ = nlp_controller.generate_chat_answer(
            query=chat_request.text,
            retrieved_documents=[],
            prior_history=prior_history,
            system_prompt=greeting_system,
        )
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_SUCCESS.value,
                "message": answer or "Hello! How can I help you plan your trip?",
                "answer": answer or "Hello! How can I help you plan your trip?",
                "sources": [],
            }
        )

    results = nlp_controller.chat(
        query=chat_request.text,
        limit=chat_request.limit,
    )

    if not results:
        fallback_prompt = (
            "You are a helpful travel assistant. "
            "No documents were found that match the user's question. "
            "Politely let the user know you couldn't find specific information about their query, "
            "and suggest they try asking about a different destination or topic (e.g., cities, attractions, hotels, restaurants, shopping). "
            "Respond in the same language as the user. Keep it brief and helpful."
        )
        answer, full_prompt, _ = nlp_controller.generate_chat_answer(
            query=chat_request.text,
            retrieved_documents=[],
            prior_history=prior_history,
            system_prompt=fallback_prompt,
        )
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_SUCCESS.value,
                "message": answer or "I couldn't find any relevant information on that topic. Try asking about a different destination!",
                "answer": answer or "I couldn't find any relevant information on that topic. Try asking about a different destination!",
                "sources": [],
            }
        )

    sources = [
        {
            "document_name": os.path.basename(doc.metadata.get("source", "")) if doc.metadata and doc.metadata.get("source") else "",
            "city": doc.metadata.get("city") if doc.metadata else None,
            "doc_type": doc.metadata.get("doc_type") if doc.metadata else None,
            "score": round(doc.score, 4),
            "excerpt": doc.text[:300],
        }
        for doc in results
    ]

    answer, full_prompt, _ = nlp_controller.generate_chat_answer(
        query=chat_request.text,
        retrieved_documents=results,
        prior_history=prior_history
    )

    if not answer:
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_ERROR.value,
                "message": "I encountered an error while generating a response. Please try again.",
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.CHAT_SUCCESS.value,
            "message": "Here's what I found based on the available documents.",
            "answer": answer,
            "sources": sources,
        }
    )
