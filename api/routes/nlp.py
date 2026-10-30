from helpers.config import get_settings
from helpers.rate_limiter import check_rate_limit
import uuid
from helpers.security import verify_admin_access
from fastapi import FastAPI, APIRouter, status, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from routes.schemes.nlp import ChatRequest, SyncActivityRequest
from controllers import NLPController
from models import ResponseSignal, ConversationModel
import os
from zoneinfo import ZoneInfo
import logging
import json
import asyncio
import urllib.request
import traceback

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


@nlp_router.post("/conversation")
async def create_conversation(request: Request):
    conversation_model = await ConversationModel.create_instance(
        db_client=request.app.db_client
    )
    conv = await conversation_model.create_conversation()
    return JSONResponse(content={"conversation_id": conv.id})


@nlp_router.get("/conversation/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str):
    conversation_model = await ConversationModel.create_instance(
        db_client=request.app.db_client
    )
    conv = await conversation_model.get_conversation(conversation_id)
    if not conv:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "conversation_not_found"}
        )
    return {
        "conversation_id": conv.id,
        "history": conv.history,
        "created_at": conv.created_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Africa/Cairo")).isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Africa/Cairo")).isoformat() if conv.updated_at else None,
    }


@nlp_router.get("/conversations")
async def list_conversations(request: Request, ids: str = Query("")):
    if not ids:
        return {"conversations": []}
    conversation_model = await ConversationModel.create_instance(
        db_client=request.app.db_client
    )
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    conversations = await conversation_model.list_conversations(id_list)
    return {"conversations": conversations}


@nlp_router.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    app_settings = get_settings()

    # Rate limiting abuse protection
    is_allowed, retry_after = check_rate_limit(
        request=request,
        max_requests=app_settings.RATE_LIMIT_CHAT_REQUESTS,
        window_seconds=app_settings.RATE_LIMIT_CHAT_WINDOW_SECONDS,
        trusted_proxies=app_settings.TRUSTED_PROXIES,
        trusted_cloudflare_proxies=getattr(app_settings, "TRUSTED_CLOUDFLARE_PROXIES", None),
        redis_url=app_settings.REDIS_URL,
        require_redis=getattr(app_settings, "REQUIRE_REDIS_RATE_LIMITER", False),
        prefix="chat"
    )
    if not is_allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "signal": "rate_limit_exceeded",
                "message": "Too many requests. Please wait a moment before sending another message.",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)}
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    text = chat_request.text.strip() if chat_request.text else ""
    if not text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.INVALID_QUERY.value,
                "message": "Query text cannot be empty.",
            }
        )

    if len(text) > app_settings.CHAT_MAX_MESSAGE_LENGTH:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "message_too_long",
                "message": f"Query exceeds maximum allowed length of {app_settings.CHAT_MAX_MESSAGE_LENGTH} characters.",
            }
        )

    prior_history = []
    conversation_model = None
    if chat_request.conversation_id:
        conversation_model = await ConversationModel.create_instance(
            db_client=request.app.db_client
        )
        conv = await conversation_model.get_conversation(chat_request.conversation_id)
        if conv:
            prior_history = conv.history

    if nlp_controller.is_greeting_query(text):
        is_ar = any('\u0600' <= c <= '\u06ff' for c in text)
        if is_ar:
            answer = "أهلاً بك في NileGuide! كيف يمكنني مساعدتك في استكشاف مصر اليوم؟"
        else:
            answer = "Hello and welcome to NileGuide! How can I help you explore Egypt today?"

        if conversation_model and chat_request.conversation_id:
            await conversation_model.append_turn(
                conversation_id=chat_request.conversation_id,
                user_msg=text,
                assistant_msg=answer,
                assistant_sources=[]
            )
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_SUCCESS.value,
                "message": answer,
                "answer": answer,
                "sources": [],
            }
        )

    try:
        results = nlp_controller.chat(
            query=text,
            limit=chat_request.limit,
            prior_history=prior_history,
            city=chat_request.city,
            category=chat_request.category,
        )
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f"Error during nlp_controller.chat [error_id={error_id}]: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "signal": ResponseSignal.CHAT_ERROR.value,
                "error": "chat_retrieval_failed",
                "message": "An error occurred while retrieving travel knowledge. Please try again later.",
                "error_id": error_id
            }
        )

    if not results:
        fallback_prompt = (
            "You are NileGuide's expert travel assistant for Egypt and world tourism. "
            "Provide a comprehensive, helpful, and concise answer to the user's travel question. "
            "Always respond in the exact same language as the user (English if asked in English, Arabic if asked in Arabic). "
            "Keep your tone welcoming and informative."
        )
        try:
            answer, full_prompt, _ = nlp_controller.generate_chat_answer(
                query=text,
                retrieved_documents=[],
                prior_history=prior_history,
                system_prompt=fallback_prompt,
            )
        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"Error during fallback generation [error_id={error_id}]: {traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={
                    "signal": ResponseSignal.CHAT_ERROR.value,
                    "error": "fallback_generation_failed",
                    "message": "An error occurred while generating travel advice. Please try again later.",
                    "error_id": error_id
                }
            )
        answer = answer or "I couldn't find any relevant information on that topic. Try asking about a different destination!"
        if conversation_model and chat_request.conversation_id:
            await conversation_model.append_turn(
                conversation_id=chat_request.conversation_id,
                user_msg=text,
                assistant_msg=answer,
                assistant_sources=[]
            )
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_SUCCESS.value,
                "message": answer,
                "answer": answer,
                "sources": [],
            }
        )

    sources = [
        {
            "activity_id": doc.metadata.get("activity_id") if doc.metadata else None,
            "title": doc.metadata.get("title") or (os.path.basename(doc.metadata.get("source", "")) if doc.metadata else ""),
            "city": doc.metadata.get("city") if doc.metadata else None,
            "category": doc.metadata.get("category") if doc.metadata else None,
            "price": doc.metadata.get("price") if doc.metadata else None,
            "image_url": doc.metadata.get("image_url") if doc.metadata else None,
            "link": doc.metadata.get("link") or (f"https://www.nileguide.online/activities/{doc.metadata.get('activity_id')}" if doc.metadata and doc.metadata.get("activity_id") else None),
            "score": round(doc.score, 4),
        }
        for doc in results
    ]

    try:
        answer, full_prompt, _ = nlp_controller.generate_chat_answer(
            query=text,
            retrieved_documents=results,
            prior_history=prior_history
        )
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f"Error during generate_chat_answer [error_id={error_id}]: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "signal": ResponseSignal.CHAT_ERROR.value,
                "error": "chat_generation_failed",
                "message": "An error occurred while generating travel response. Please try again later.",
                "error_id": error_id
            }
        )

    if not answer:
        return JSONResponse(
            content={
                "signal": ResponseSignal.CHAT_ERROR.value,
                "message": "I encountered an error while generating a response. Please try again.",
            }
        )

    if conversation_model and chat_request.conversation_id:
        await conversation_model.append_turn(
            conversation_id=chat_request.conversation_id,
            user_msg=text,
            assistant_msg=answer,
            assistant_sources=sources[:3],
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.CHAT_SUCCESS.value,
            "message": "Here's what I found based on the available documents.",
            "answer": answer,
            "sources": sources,
        }
    )


@nlp_router.post("/chat/stream")
async def chat_stream_endpoint(request: Request, chat_request: ChatRequest):
    app_settings = get_settings()

    is_allowed, retry_after = check_rate_limit(
        request=request,
        max_requests=app_settings.RATE_LIMIT_STREAM_REQUESTS,
        window_seconds=app_settings.RATE_LIMIT_STREAM_WINDOW_SECONDS,
        trusted_proxies=app_settings.TRUSTED_PROXIES,
        trusted_cloudflare_proxies=getattr(app_settings, "TRUSTED_CLOUDFLARE_PROXIES", None),
        redis_url=app_settings.REDIS_URL,
        require_redis=getattr(app_settings, "REQUIRE_REDIS_RATE_LIMITER", False),
        prefix="stream"
    )
    if not is_allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "signal": "rate_limit_exceeded",
                "message": "Too many streaming requests. Please wait a moment.",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)}
        )
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    text = chat_request.text.strip() if chat_request.text else ""
    if not text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.INVALID_QUERY.value}
        )

    prior_history = []
    conversation_model = None
    if chat_request.conversation_id:
        conversation_model = await ConversationModel.create_instance(
            db_client=request.app.db_client
        )
        conv = await conversation_model.get_conversation(chat_request.conversation_id)
        if conv:
            prior_history = conv.history

    results = nlp_controller.chat(
        query=text,
        limit=chat_request.limit,
        prior_history=prior_history,
        city=chat_request.city,
        category=chat_request.category,
    ) or []

    sources = [
        {
            "activity_id": doc.metadata.get("activity_id") if doc.metadata else None,
            "title": doc.metadata.get("title") or (os.path.basename(doc.metadata.get("source", "")) if doc.metadata else ""),
            "city": doc.metadata.get("city") if doc.metadata else None,
            "category": doc.metadata.get("category") if doc.metadata else None,
            "price": doc.metadata.get("price") if doc.metadata else None,
            "image_url": doc.metadata.get("image_url") if doc.metadata else None,
            "link": doc.metadata.get("link") or (f"https://www.nileguide.online/activities/{doc.metadata.get('activity_id')}" if doc.metadata and doc.metadata.get("activity_id") else None),
            "score": round(doc.score, 4),
        }
        for doc in results
    ]

    answer, _, _ = nlp_controller.generate_chat_answer(
        query=text,
        retrieved_documents=results,
        prior_history=prior_history,
    )
    answer = answer or "No results found."

    if conversation_model and chat_request.conversation_id:
        await conversation_model.append_turn(
            conversation_id=chat_request.conversation_id,
            user_msg=text,
            assistant_msg=answer,
            assistant_sources=sources[:3],
        )

    async def event_generator():
        yield f"event: sources\ndata: {json.dumps(sources[:3])}\n\n"
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.015)
        yield f"event: done\ndata: {json.dumps({'answer': answer})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@nlp_router.post("/sync/activity", dependencies=[Depends(verify_admin_access)])
async def sync_activity_endpoint(request: Request, sync_req: SyncActivityRequest):
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    if sync_req.action == "delete" or not sync_req.is_active:
        res = nlp_controller.delete_activity(sync_req.activity_id)
        return JSONResponse(content={"signal": "activity_synced", "action": "deleted", "activity_id": sync_req.activity_id})

    res = nlp_controller.sync_activity(sync_req.dict())
    return JSONResponse(content={"signal": "activity_synced", "action": "upserted", "activity_id": sync_req.activity_id})


@nlp_router.post("/sync/reconcile", dependencies=[Depends(verify_admin_access)])
async def reconcile_activities_endpoint(request: Request):
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    activities = []
    for page in [1, 2]:
        api_url = f"https://nileguide.runasp.net/api/Activities?Page={page}&PageSize=50"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data") or data.get("items") or data
                activities.extend(items)
        except Exception as e:
            logger.error(f"Failed to fetch activities page {page}: {e}")

    synced_count = 0
    for act in activities:
        try:
            act_id = act.get("activityID")
            if not act_id:
                continue
            nlp_controller.sync_activity({
                "activity_id": act_id,
                "title": act.get("activityName", ""),
                "city": act.get("cityName", "Egypt"),
                "category": act.get("categoryName", "General"),
                "price": f"{act.get('minPrice', act.get('price', 0))} {act.get('priceCurrency', 'USD')}",
                "description": act.get("description", ""),
                "image_url": act.get("imageUrl", (act.get("images") or [""])[0] if act.get("images") else ""),
                "link": f"https://www.nileguide.online/activities/{act_id}",
                "is_active": True,
            })
            synced_count += 1
        except Exception as e:
            logger.warning(f"Error syncing activity {act.get('activityID')}: {e}")

    return JSONResponse(content={"signal": "reconcile_success", "total_synced": synced_count})
