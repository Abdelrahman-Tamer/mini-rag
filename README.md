# Full RAG — Multi-City Tourism Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for tourism insights. Users ask questions about cities, attractions, restaurants, hotels, and shopping — the system retrieves relevant document chunks from Qdrant and generates answers grounded in uploaded content via OpenAI's `gpt-4o-mini`.

Conversations are **stateful** — history is persisted in MongoDB, enabling follow-up questions, query rewriting, and a natural chat experience.

---

## Architecture

```
Angular (Frontend)
    ↓  HTTP
.NET API (Auth + User → Conversation Mapping)
    ↓  HTTP
FastAPI RAG Server (port 5000)
    ├── OpenAI       gpt-4o-mini (generation) + text-embedding-3-small (embeddings)
    ├── Qdrant Cloud tourism_knowledge_base (single collection, metadata: city/doc_type/asset_id)
    └── MongoDB      conversations, chunks, assets
```

- **Angular** renders the chat UI
- **.NET** handles authentication and maps users to `conversation_id`s
- **RAG** owns all conversation history, document storage, and LLM orchestration

---

## Features

- **Stateful Conversations** — history stored in MongoDB, loaded by `conversation_id`
- **Query Rewriting** — follow-up questions ("what about hotels there?") are rewritten into standalone search queries using the last 5 turns of context
- **Auto-Titling** — first user message automatically becomes the conversation title
- **Cairo Timezone** — all timestamps returned in `Africa/Cairo` timezone
- **Batch File Upload** — upload multiple PDFs/DOCXs/CSVs/TXTs at once, auto-chunked at 500 chars with 20 overlap
- **Greeting Detection** — pure greetings skip vector search and return a warm welcome
- **Source Citations** — every answer includes sources with document name, city, type, score, and excerpt
- **Single Collection** — all chunks in one Qdrant collection with metadata, no per-city routing

---

## Quick Start

### Prerequisites

- Python 3.10+
- MongoDB Atlas (or local MongoDB)
- Qdrant Cloud (or local Qdrant)
- OpenAI API key

### Setup

```bash
# Clone
git clone https://github.com/sultanitos1/full-rag
cd full-rag/src

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, MONGODB_URL, QDRANT_DB_URL, QDRANT_DB_API_KEY

# Run
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Deploy on EC2

```bash
# Systemd service: rag.service
# Deploy script (~/deploy.sh):
git pull && pip install -r requirements.txt -q && sudo systemctl restart rag
```

---

## API Reference

### Health Check

```
GET /api/v1/
```
```json
{ "app_name": "Mini RAG", "app_version": "0.1.0" }
```

### Conversations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/nlp/conversation` | Create a new conversation → `conversation_id` |
| GET | `/api/v1/nlp/conversation/{id}` | Get full history + timestamps |
| GET | `/api/v1/nlp/conversations?ids=a,b,c` | List titles + `updated_at` for sidebar |

**POST /api/v1/nlp/conversation**
```json
{ "conversation_id": "abc123..." }
```

**GET /api/v1/nlp/conversation/abc123**
```json
{
  "conversation_id": "abc123",
  "history": [
    { "role": "user", "content": "What hotels are in Cairo?" },
    { "role": "assistant", "content": "Cairo has many hotels including..." }
  ],
  "created_at": "2026-06-16T03:09:55+02:00",
  "updated_at": "2026-06-16T03:09:55+02:00"
}
```

### Chat

```
POST /api/v1/nlp/chat
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | User question |
| `conversation_id` | string | no | — | Conversation to continue (creates new if omitted) |
| `limit` | int | no | 5 | Max sources to retrieve |

```json
{
  "text": "What hotels are in Luxor?",
  "conversation_id": "abc123",
  "limit": 5
}
```

**Response:**
```json
{
  "signal": "chat_success",
  "message": "Here's what I found based on the available documents.",
  "answer": "Luxor has several highly-rated hotels including the Sofitel Winter Palace...",
  "sources": [
    {
      "document_name": "luxor_hotels_guide.pdf",
      "city": "luxor",
      "doc_type": "hotels",
      "score": 0.8921,
      "excerpt": "The Sofitel Winter Palace Luxor is a historic hotel..."
    }
  ]
}
```

Chat flow:
1. Load history from MongoDB by `conversation_id`
2. If greeting → respond directly, skip retrieval
3. Else: rewrite query with last 5 turns → embed → Qdrant search → filter by score ≥ 0.4
4. Send docs + last 5 turns to `gpt-4o-mini` → generate answer
5. Save user + assistant messages to MongoDB
6. Return `{ answer, sources }`

### Data Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/data/multi-upload` | Upload multiple files (batch) |
| GET | `/api/v1/data/files` | List all uploaded files |
| PUT | `/api/v1/data/file/{file_id}` | Replace a file and re-index |
| DELETE | `/api/v1/data/file/{file_id}` | Delete file, chunks, and vectors |

#### POST /api/v1/data/multi-upload

Multipart form data:
- `files` — one or more files (PDF, DOCX, CSV, TXT)
- `chunk_size` — optional (default 500)
- `overlap_size` — optional (default 20)

All files are uploaded without city/doc_type metadata. Metadata is attached at the chunk level via `chunk_metadata.asset_id`.

---

## Environment Variables

All config is in `.env`. See `.env.example` for a template.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_API_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `GENERATION_MODEL_ID` | No | `gpt-4o-mini` | Model for answer generation |
| `EMBEDDING_MODEL_ID` | No | `text-embedding-3-small` | Model for embeddings |
| `EMBEDDING_MODEL_SIZE` | No | `1536` | Embedding dimension |
| `MONGODB_URL` | Yes | — | MongoDB connection string |
| `MONGODB_DATABASE` | No | `mini-rag` | MongoDB database name |
| `QDRANT_DB_URL` | Yes | — | Qdrant cloud URL |
| `QDRANT_DB_API_KEY` | Yes | — | Qdrant API key |
| `VECTOR_DB_COLLECTION_NAME` | No | `tourism_knowledge_base` | Qdrant collection |
| `DISTANCE_METHOD` | No | `cosine` | Vector distance metric |
| `SCORE_THRESHOLD` | No | `0.4` | Minimum vector search score |
| `INPUT_DEFAULT_MAX_CHARACTERS` | No | `4096` | Max input chars for LLM |
| `GENERATION_DEFAULT_MAX_TOKENS` | No | `500` | Max output tokens |
| `GENERATION_DEFAULT_TEMPERATURE` | No | `0.1` | LLM temperature |
| `FILE_ALLOWED_TYPES` | No | `[text/plain, application/pdf, ...]` | Allowed MIME types |
| `FILE_MAX_SIZE` | No | `10` | Max file size in MB |
| `PRIMARY_LANG` | No | `en` | Prompt language |

---

## Project Structure

```
src/
├── main.py                       # FastAPI entry point, mounts routers, initializes clients
├── .env                          # Environment config (gitignored)
├── .env.example                  # Environment template (committed)
├── requirements.txt              # Python dependencies
│
├── routes/                       # HTTP layer
│   ├── base.py                   # GET /api/v1/ health check
│   ├── data.py                   # File upload/list/update/delete endpoints
│   └── nlp.py                    # Conversation CRUD + chat endpoint
│
├── controllers/                  # Business logic
│   ├── BaseController.py         # Shared base (app_settings, file paths)
│   ├── DataController.py         # File validation + unique naming
│   ├── NLPController.py          # Chat, query rewrite, answer generation, indexing
│   └── ProcessController.py      # File loading + text chunking
│
├── models/                       # MongoDB data access
│   ├── BaseDataModel.py          # DB client + settings
│   ├── AssetModel.py             # File asset CRUD
│   ├── ChunkModel.py             # Text chunk CRUD
│   ├── ConversationModel.py      # Conversation + history CRUD
│   └── db_schemes/               # Pydantic schemas (Asset, Conversation, DataChunk)
│
├── stores/                       # External service abstractions
│   ├── llm/                      # OpenAI provider + template parser
│   └── vectordb/                 # Qdrant provider
│
├── helpers/config.py             # Settings loader from .env
├── assets/files/                 # Uploaded files on disk (gitignored)
└── tests/                        # Manual test payloads
```

---

## License

MIT
