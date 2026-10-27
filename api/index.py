from fastapi import FastAPI, Request
from routes import base, data, nlp
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware  
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from fastapi.responses import JSONResponse
import logging
import traceback

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="NileGuide AI Chatbot API")

async def startup_span():
    if getattr(app.state, "initialized", False):
        return
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                            embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # vector db client
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
    return {
        "status": "ok",
        "service": "NileGuide AI Chatbot API",
        "endpoints": {
            "chat": "/api/v1/nlp/chat",
            "conversation": "/api/v1/nlp/conversation",
            "welcome": "/api/v1"
        }
    }

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
