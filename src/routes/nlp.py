from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ChatRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from controllers import NLPController
from controllers.NLPController import load_city_mappings
from models import ResponseSignal
import os

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=getattr(request.app, 'rerank_client', None),
        rerank_model_id=getattr(request.app, 'rerank_model_id', None),
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    while has_records:
        page_chunks = await chunk_model.get_project_chunks(project_id=project.id, page_no=page_no)
        if len(page_chunks):
            page_no += 1

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)

        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )

        inserted_items_count += len(page_chunks)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=getattr(request.app, 'rerank_client', None),
        rerank_model_id=getattr(request.app, 'rerank_model_id', None),
    )

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=getattr(request.app, 'rerank_client', None),
        rerank_model_id=getattr(request.app, 'rerank_model_id', None),
    )

    results = nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [result.dict() for result in results]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=getattr(request.app, 'rerank_client', None),
        rerank_model_id=getattr(request.app, 'rerank_model_id', None),
    )

    answer, full_prompt, chat_history = nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )


@nlp_router.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=getattr(request.app, 'rerank_client', None),
        rerank_model_id=getattr(request.app, 'rerank_model_id', None),
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
                "related_questions": [],
            }
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    db_cities = await asset_model.collection.distinct(
        "asset_config.city",
        {"asset_config.city": {"$nin": [None, "unknown"]}}
    )
    cfg = load_city_mappings()
    known_cities = list(set((db_cities or []) + (cfg.get("known_cities", []) or [])))
    detected_city = nlp_controller.detect_city_from_query(chat_request.text, known_cities)

    results = nlp_controller.chat(
        query=chat_request.text,
        limit=chat_request.limit,
        city=detected_city,
        doc_type=chat_request.doc_type
    )

    if not results:
        return JSONResponse(
            content={
                "signal": ResponseSignal.NO_RELEVANT_DOCUMENTS.value,
                "message": "I couldn't find any relevant information on that topic. Try asking about a different destination!",
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

    related_questions = nlp_controller.generate_related_questions(
        query=chat_request.text,
        answer=answer
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.CHAT_SUCCESS.value,
            "message": "Here's what I found based on the available documents.",
            "answer": answer,
            "sources": sources,
            "related_questions": related_questions,
        }
    )
