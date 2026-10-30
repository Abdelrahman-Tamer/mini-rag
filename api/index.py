import os
import sys
from pathlib import Path

# Add api directory to sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.responses import JSONResponse
import logging
import traceback

logger = logging.getLogger("uvicorn.error")

# Top-level ASGI entrypoint (REQUIRED at indentation 0 for Vercel)
app = FastAPI(title="NileGuide AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def restore_path_middleware(request: Request, call_next):
    route = request.query_params.get("_route")
    if route:
        while route.startswith("//"):
            route = route[1:]
        request.scope["path"] = route
    return await call_next(request)

async def init_services():
    if getattr(app.state, "initialized", False):
        return True
    try:
        from helpers.config import get_settings, validate_admin_security_config, validate_embedding_config
        from motor.motor_asyncio import AsyncIOMotorClient
        from stores.llm.LLMProviderFactory import LLMProviderFactory
        from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
        from stores.llm.templates.template_parser import TemplateParser

        settings = get_settings()
        validate_admin_security_config(settings)
        validate_embedding_config(settings)

        if settings.MONGODB_URL:
            app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
            app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
        else:
            app.mongo_conn = None
            app.db_client = None

        llm_provider_factory = LLMProviderFactory(settings)
        vectordb_provider_factory = VectorDBProviderFactory(settings)

        # generation client
        app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
        app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

        # embedding client
        app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE
        )
        
        # vector db client
        if settings.QDRANT_DB_URL:
            app.vectordb_client = vectordb_provider_factory.create(
                provider=settings.VECTOR_DB_BACKEND
            )
            app.vectordb_client.connect()
        else:
            app.vectordb_client = None

        app.template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )
        app.state.initialized = True
        logger.info("NileGuide AI Services successfully initialized.")
        return True
    except Exception as e:
        logger.error(f"Service initialization error: {traceback.format_exc()}")
        app.state.init_error = traceback.format_exc()
        return False

@app.middleware("http")
async def ensure_initialized_middleware(request: Request, call_next):
    if not getattr(app.state, "initialized", False):
        success = await init_services()
        if not success and request.url.path.startswith("/api/v1/nlp"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Service Initialization Failed",
                    "details": getattr(app.state, "init_error", "Unknown error")
                }
            )
    response = await call_next(request)
    return response

async def shutdown_span():
    if hasattr(app, "mongo_conn") and app.mongo_conn:
        app.mongo_conn.close()
    if hasattr(app, "vectordb_client") and app.vectordb_client:
        app.vectordb_client.disconnect()

app.on_event("startup")(init_services)
app.on_event("shutdown")(shutdown_span)

@app.get("/")
@app.get("/api")
async def root():
    from helpers.config import get_settings
    settings = get_settings()
    return {
        "status": "ok",
        "service": "NileGuide AI Chatbot API",
        "config": {
            "mongodb_configured": bool(settings.MONGODB_URL),
            "qdrant_configured": bool(settings.QDRANT_DB_URL),
            "openai_key_configured": bool(settings.OPENAI_API_KEY)
        },
        "endpoints": {
            "chat": "/api/v1/nlp/chat",
            "conversation": "/api/v1/nlp/conversation",
            "welcome": "/api/v1"
        }
    }

# Include routers
from routes import base, data, nlp
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
