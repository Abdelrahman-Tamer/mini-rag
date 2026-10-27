---
title: NileGuide RAG
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# ðŸ›ï¸ Full RAG â€” Multi-City Tourism Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?style=for-the-badge&logo=openai" alt="OpenAI">
  <img src="https://img.shields.io/badge/Qdrant-Cloud-FF6F00?style=for-the-badge&logo=qdrant" alt="Qdrant">
  <img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb" alt="MongoDB">
  <img src="https://img.shields.io/badge/Angular-17-DD0031?style=for-the-badge&logo=angular" alt="Angular">
  <img src="https://img.shields.io/badge/.NET-8-512BD4?style=for-the-badge&logo=dotnet" alt=".NET">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>A production-ready Retrieval-Augmented Generation chatbot for tourism insights.</b><br>
  Users ask about cities, attractions, restaurants, hotels, and shopping â€”<br>
  the system retrieves relevant document chunks and generates grounded answers via OpenAI.
</p>

---

## âœ¨ Features

| Feature | Description |
|---------|-------------|
| ðŸ§  **Stateful Conversations** | History persisted in MongoDB â€” resume any chat, anywhere |
| ðŸ”„ **Query Rewriting** | Follow-ups like "what about hotels there?" â†’ standalone search query |
| ðŸ·ï¸ **Auto-Titling** | First user message becomes the conversation title â€” sidebar ready |
| ðŸ“„ **Multi-File Upload** | Batch upload PDF/DOCX/CSV/TXT â€” auto-chunked at 500 chars |
| ðŸ™ï¸ **Single Collection** | All cities share one Qdrant collection â€” metadata-based filtering |
| ðŸŒ **Cairo Timezone** | All timestamps in `Africa/Cairo` â€” Egypt tourism focus |
| ðŸ‘‹ **Greeting Detection** | "Hi", "Hello", "Ù…Ø±Ø­Ø¨Ø§" â†’ warm welcome, no vector search |
| ðŸ“Ž **Source Citations** | Every answer includes document name, city, score, and excerpt |

---

## ðŸ—ï¸ Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                       Angular Frontend                          â”‚
â”‚          Chat UI + Conversation Sidebar + File Manager          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                             â”‚ HTTP
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                         .NET API                                â”‚
â”‚          Authentication Â· User â†’ Conversation Mapping           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                             â”‚ HTTP
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      FastAPI RAG Server                         â”‚
â”‚                                                                 â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚      Routes          â”‚   â”‚         Controllers            â”‚  â”‚
â”‚  â”‚  Â· Health (/)        â”‚   â”‚  Â· DataController â€” validation â”‚  â”‚
â”‚  â”‚  Â· Data (/data)      â”‚   â”‚  Â· ProcessController â€” chunkingâ”‚  â”‚
â”‚  â”‚  Â· Chat (/nlp)       â”‚   â”‚  Â· NLPController â€” RAG engine  â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                                 â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚       Models         â”‚   â”‚          Stores                â”‚  â”‚
â”‚  â”‚  Â· AssetModel        â”‚   â”‚  Â· OpenAIProvider (gpt-4o-mini)â”‚  â”‚
â”‚  â”‚  Â· ChunkModel        â”‚   â”‚  Â· OpenAIProvider (embeddings) â”‚  â”‚
â”‚  â”‚  Â· ConversationModel â”‚   â”‚  Â· QdrantDBProvider            â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                                 â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚  â”‚   MongoDB    â”‚  â”‚   Qdrant Cloud â”‚  â”‚   OpenAI API       â”‚   â”‚
â”‚  â”‚   Atlas      â”‚  â”‚   vectors      â”‚  â”‚   generation + emb â”‚   â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Data flow**: `Angular UI â†” .NET Auth â†” FastAPI RAG â†” OpenAI + Qdrant + MongoDB`

---

## ðŸš€ Quick Start

### Prerequisites

- Python 3.10+
- [MongoDB Atlas](https://www.mongodb.com/atlas) (free tier works)
- [Qdrant Cloud](https://cloud.qdrant.io/) (free 1GB cluster)
- [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone
git clone https://github.com/sultanitos1/full-rag
cd full-rag/src

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

Edit `.env` with your keys:

```ini
OPENAI_API_KEY="sk-..."                    # Your OpenAI API key
MONGODB_URL="mongodb+srv://..."            # Your MongoDB Atlas URI
QDRANT_DB_URL="https://..."                # Your Qdrant cloud URL
QDRANT_DB_API_KEY="eyJ..."                 # Your Qdrant API key
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Deploy on EC2

```bash
# ~/deploy.sh
git pull && pip install -r requirements.txt -q && sudo systemctl restart rag
```

---

## ðŸ“¡ API Reference

### Health Check

```
GET /api/v1/
```

```json
{ "app_name": "Mini RAG", "app_version": "0.1.0" }
```

---

### ðŸ’¬ Conversations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/nlp/conversation` | Create new chat session |
| `GET`  | `/api/v1/nlp/conversation/{id}` | Get full history |
| `GET`  | `/api/v1/nlp/conversations?ids=a,b,c` | List titles for sidebar |

**POST /api/v1/nlp/conversation**
```json
// Response
{ "conversation_id": "a1b2c3d4e5f6a7b8c9d0e1f2" }
```

**GET /api/v1/nlp/conversation/a1b2c3d4e5f6a7b8c9d0e1f2**
```json
{
  "conversation_id": "a1b2c3d4e5f6a7b8c9d0e1f2",
  "history": [
    { "role": "user", "content": "What hotels are in Cairo?" },
    { "role": "assistant", "content": "Cairo has many hotels..." }
  ],
  "created_at": "2026-06-16T03:09:55+02:00",
  "updated_at": "2026-06-16T03:09:55+02:00"
}
```

---

### ðŸ¤– Chat

```
POST /api/v1/nlp/chat
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | âœ… | â€” | User's question |
| `conversation_id` | `string` | âŒ | `null` | Existing conversation to continue |
| `limit` | `int` | âŒ | `5` | Max sources to retrieve |

```json
{
  "text": "What hotels are in Luxor?",
  "conversation_id": "a1b2c3d4e5f6a7b8c9d0e1f2",
  "limit": 5
}
```

**Response:**
```json
{
  "signal": "chat_success",
  "message": "Here's what I found based on the available documents.",
  "answer": "Luxor has several highly-rated hotels including the Sofitel Winter Palace Luxor, a historic hotel on the Nile celebrating its 100th anniversary.",
  "sources": [
    {
      "document_name": "luxor_hotels_guide.pdf",
      "city": "unknown",
      "doc_type": "general",
      "score": 0.8921,
      "excerpt": "Sofitel Winter Palace Luxor is a 5-star hotel..."
    }
  ]
}
```

---

### ðŸ“ File Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/data/multi-upload` | Upload multiple files (batch) |
| `GET`  | `/api/v1/data/files` | List all uploaded files |
| `PUT`  | `/api/v1/data/file/{file_id}` | Replace a file + re-index |
| `DELETE` | `/api/v1/data/file/{file_id}` | Delete file, chunks, and vectors |

**POST /api/v1/data/multi-upload**

Multipart form:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `files` | `UploadFile[]` | âœ… | â€” | PDF, DOCX, CSV, or TXT files |
| `chunk_size` | `int` | âŒ | `500` | Characters per chunk |
| `overlap_size` | `int` | âŒ | `20` | Chunk overlap characters |

```json
// Response
{
  "signal": "multi_upload_success",
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "results": [
    { "file_name": "luxor_hotels.pdf", "status": "success", "file_id": "abc...", "inserted_chunks": 12 }
  ]
}
```

---

## âš™ï¸ Configuration

All configuration is via `.env`. See [`.env.example`](src/.env.example) for a template.

### Required Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `MONGODB_URL` | MongoDB Atlas connection string |
| `QDRANT_DB_URL` | Qdrant cloud URL |
| `QDRANT_DB_API_KEY` | Qdrant API key |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_MODEL_ID` | `gpt-4o-mini` | Model for answer generation |
| `EMBEDDING_MODEL_ID` | `text-embedding-3-small` | Model for embeddings |
| `EMBEDDING_MODEL_SIZE` | `1536` | Embedding dimension |
| `SCORE_THRESHOLD` | `0.4` | Minimum vector search score |
| `INPUT_DEFAULT_MAX_CHARACTERS` | `4096` | Max input chars to LLM |
| `GENERATION_DEFAULT_MAX_TOKENS` | `500` | Max output tokens |
| `GENERATION_DEFAULT_TEMPERATURE` | `0.1` | LLM temperature |
| `VECTOR_DB_COLLECTION_NAME` | `tourism_knowledge_base` | Qdrant collection name |
| `DISTANCE_METHOD` | `cosine` | Vector distance metric |
| `FILE_MAX_SIZE` | `10` | Max file size in MB |
| `PRIMARY_LANG` | `en` | Prompt language |
| `DEFAULT_CITY` | `unknown` | Default city for uploads |
| `DEFAULT_DOC_TYPE` | `general` | Default doc type for uploads |

---

## ðŸ“‚ Project Structure

```
src/
â”œâ”€â”€ main.py                       # FastAPI entry point
â”œâ”€â”€ .env.example                  # Configuration template
â”‚
â”œâ”€â”€ routes/                       # HTTP layer
â”‚   â”œâ”€â”€ base.py                   # GET /api/v1/ â€” health check
â”‚   â”œâ”€â”€ data.py                   # File upload/list/update/delete
â”‚   â””â”€â”€ nlp.py                    # Conversation CRUD + chat
â”‚
â”œâ”€â”€ controllers/                  # Business logic
â”‚   â”œâ”€â”€ BaseController.py         # Shared base
â”‚   â”œâ”€â”€ DataController.py         # File validation
â”‚   â”œâ”€â”€ ProcessController.py      # Document loading & chunking
â”‚   â””â”€â”€ NLPController.py          # RAG engine (chat, search, index)
â”‚
â”œâ”€â”€ models/                       # MongoDB data access
â”‚   â”œâ”€â”€ AssetModel.py             # File metadata CRUD
â”‚   â”œâ”€â”€ ChunkModel.py             # Text chunk CRUD
â”‚   â”œâ”€â”€ ConversationModel.py      # Chat history CRUD
â”‚   â””â”€â”€ db_schemes/               # Pydantic schemas
â”‚
â”œâ”€â”€ stores/                       # External service abstractions
â”‚   â”œâ”€â”€ llm/                      # OpenAI provider + prompt templates
â”‚   â””â”€â”€ vectordb/                 # Qdrant provider
â”‚
â”œâ”€â”€ helpers/config.py             # Settings loader from .env
â””â”€â”€ assets/files/                 # Uploaded files on disk
```

For a complete deep-dive into every file, function, and endpoint, see [**ARCHITECTURE.md**](ARCHITECTURE.md).

---

## ðŸ”„ Chat Flow

```
User: "What hotels are in Luxor?"
        â”‚
        â–¼
  1. Load conversation history from MongoDB
        â”‚
        â–¼
  2. Is it a greeting? â”€â”€â”€ Yes â†’ Warm welcome (skip search)
        â”‚ No
        â–¼
  3. Rewrite query with last 5 turns of context
     "What hotels are in Luxor?" â†’ "What hotels are in Luxor?"
        â”‚
        â–¼
  4. Embed â†’ Qdrant search â†’ filter by score â‰¥ 0.4
        â”‚
        â–¼
  5. Send docs + last 5 turns to GPT-4o-mini
        â”‚
        â–¼
  6. Save Q&A to MongoDB
        â”‚
        â–¼
  7. Return { answer, sources }
```

**Why this works better than a naive RAG:**
- ðŸ”„ **Query rewriting** handles follow-ups naturally ("what about there?" â†’ "hotels in Luxor")
- ðŸ§¹ **History truncation** keeps latency predictable as conversations grow
- ðŸ“ **Budget trimming** prevents silent prompt truncation by the LLM
- ðŸ—„ï¸ **Persistent MongoDB history** survives server restarts

---

## ðŸ› ï¸ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | FastAPI (Python) | Async API server |
| **LLM** | OpenAI gpt-4o-mini | Answer generation |
| **Embeddings** | OpenAI text-embedding-3-small | Vector embeddings (1536d) |
| **Vector DB** | Qdrant Cloud | Similarity search |
| **Database** | MongoDB Atlas | Conversation history, assets, chunks |
| **Frontend** | Angular 17 | Chat UI |
| **Middleware** | .NET 8 | Authentication, user mapping |
| **Deployment** | Ubuntu EC2 + systemd | Production server |
| **File parsing** | LangChain (PyMuPDF, etc.) | PDF/DOCX/CSV/TXT loading |

---

## ðŸ“– Documentation

| Document | Description |
|----------|-------------|
| [`README.md`](README.md) | This file â€” quick start + overview |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Complete architecture â€” every file, endpoint, function explained |
| [`CODEBASE_CONSCIOUSNESS.md`](CODEBASE_CONSCIOUSNESS.md) | Internal dev reference â€” bugs, dead code, recent changes |

---

## ðŸ¤ Contributing

This is a graduation project. Contributions, issues, and feature requests are welcome!

---

## ðŸ“„ License

[MIT](LICENSE)
