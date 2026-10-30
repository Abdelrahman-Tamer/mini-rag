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

try:
    from routes import base, data, nlp
    from motor.motor_asyncio import AsyncIOMotorClient
    from helpers.config import get_settings, validate_admin_security_config, validate_embedding_config
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
    from stores.llm.templates.template_parser import TemplateParser

    app = FastAPI(title="NileGuide AI Chatbot API")

    async def startup_span():
        if getattr(app.state, "initialized", False):
            return
        settings = get_settings()
        validate_admin_security_config(settings)
        validate_embedding_config(settings)

        if not settings.MONGODB_URL:
            logger.warning("MONGODB_URL is not set!")
            app.mongo_conn = None
            app.db_client = None
        else:
            app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
            app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

        llm_provider_factory = LLMProviderFactory(settings)
        vectordb_provider_factory = VectorDBProviderFactory(settings)

        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set!")
            app.generation_client = None
            app.embedding_client = None
        else:
            # generation client
            app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
            app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

            # embedding client
            app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
            app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                                    embedding_size=settings.EMBEDDING_MODEL_SIZE)
        
        # vector db client
        if not settings.QDRANT_DB_URL:
            logger.warning("QDRANT_DB_URL is not set!")
            app.vectordb_client = None
        else:
            app.vectordb_client = vectordb_provider_factory.create(
                provider=settings.VECTOR_DB_BACKEND
            )
            app.vectordb_client.connect()

        app.template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )
        app.state.initialized = True
        logger.info("NileGuide AI Services successfully initialized.")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def ensure_initialized_middleware(request: Request, call_next):
        if not getattr(app.state, "initialized", False):
            try:
                await startup_span()
            except Exception as e:
                err = traceback.format_exc()
                logger.error(f"Startup error: {err}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Service Initialization Failed", "details": str(e), "traceback": err}
                )
        response = await call_next(request)
        return response

    async def shutdown_span():
        if hasattr(app, "mongo_conn") and app.mongo_conn:
            app.mongo_conn.close()
        if hasattr(app, "vectordb_client") and app.vectordb_client:
            app.vectordb_client.disconnect()

    app.on_event("startup")(startup_span)
    app.on_event("shutdown")(shutdown_span)

    @app.get("/")
    async def root():
        settings = get_settings()
        validate_admin_security_config(settings)
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

    app.include_router(base.base_router)
    app.include_router(data.data_router)
    app.include_router(nlp.nlp_router)

except Exception as e:
    app = FastAPI(title="NileGuide AI Chatbot API (Fallback)")
    _startup_error = traceback.format_exc()
    logger.error(f"Failed to load application: {_startup_error}")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Application Import Failed",
                "details": str(e),
                "traceback": _startup_error
            }
        )
