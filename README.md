# MediBot &mdash; Medical Chatbot (Generative AI, RAG)

A retrieval-augmented generation (RAG) chatbot that answers medical information
questions using reference PDFs in `Data/` (currently public-domain NIH patient
guides), Pinecone for vector search, and OpenAI or Gemini for generation.

> **Note on data:** the original `Medical_book.pdf` on disk was found corrupted
> (unrecoverable text encoding damage) and moved to `Data/_corrupted_original/`.
> Its previously-indexed vectors remain live in the Pinecone index and are still
> used for retrieval, alongside the NIH guides. Any PDF dropped into `Data/` is
> picked up by `python store_index.py`.

> **⚠️ This is a portfolio/demo project, not a medical device.** MediBot is NOT a
> substitute for professional medical advice, diagnosis, or treatment. It has not
> been reviewed by a clinician and is not HIPAA/GDPR certified. See
> [Limitations & Roadmap](#limitations--roadmap) below before considering any real
> deployment.

## Architecture

```
                     ┌─────────────────────┐
                     │   Data/*.pdf         │
                     └──────────┬──────────┘
                                │  store_index.py (one-time / re-run to refresh)
                                │  load → chunk → embed
                                ▼
                     ┌─────────────────────┐
                     │  Pinecone index      │  vector store (384-dim,
                     │  ("medicalbot")      │  all-MiniLM-L6-v2 embeddings)
                     └──────────┬──────────┘
                                │  similarity search (top_k)
                                ▼
  Browser ──HTTP──▶  FastAPI app.py                     ┌───────────────────┐
  (chat.html/js)      ├─ GET  /              chat UI     │ src/rag_chain.py   │
                       ├─ GET  /health        healthcheck │  history-aware     │
                       └─ POST /api/chat ────────────────▶│  retriever + LLM   │
                                                           │  chain, disclaimer │
                                                           │  injected server-  │
                                                           │  side              │
                                                           └─────────┬─────────┘
                                                                     │
                                                     LLM_PROVIDER=openai|gemini
                                                                     ▼
                                                  ┌────────────────────────────┐
                                                  │ OpenAI (gpt-4o-mini) or     │
                                                  │ Gemini (gemini-flash-latest)│
                                                  └────────────────────────────┘
```

Conversation history is kept per `session_id` in an in-memory store
(`src/rag_chain.py::SessionHistoryStore`) so follow-up questions are
context-aware. See [Limitations](#limitations--roadmap) for why this isn't
production-durable yet.

## How to run

### Step 1: Clone and create a virtual environment

```bash
git clone https://github.com/rheagupta31/Medical-Chatbot-Generative-AI
cd Medical-Chatbot-Generative-AI
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env`:
- `PINECONE_API_KEY` &mdash; required.
- `LLM_PROVIDER` &mdash; `openai` or `gemini`.
- `OPENAI_API_KEY` and/or `GOOGLE_API_KEY` &mdash; only the one matching `LLM_PROVIDER` is required.

### Step 4: Build the vector index (run once, or whenever `Data/` changes)

```bash
python store_index.py
```

This loads every PDF in `Data/`, chunks it, embeds it with
`sentence-transformers/all-MiniLM-L6-v2`, and creates/populates the Pinecone
index named by `PINECONE_INDEX_NAME` (default `medicalbot`).

### Step 5: Run the app

```bash
./run.sh
```

(or `uvicorn app:app` without auto-reload; avoid a bare `uvicorn app:app --reload`,
which watches the entire `.venv` directory and gets stuck in a restart loop)

Open [http://localhost:8000](http://localhost:8000) for the chat UI.

### Usage example (API)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Acne?"}'
```

```json
{
  "answer": "Acne is a skin condition characterized by...",
  "disclaimer": "This information is for general educational purposes only and is NOT a substitute for professional medical advice...",
  "sources": ["Medical_book.pdf"],
  "session_id": "a1b2c3d4-..."
}
```

Pass the returned `session_id` on the next request to continue the same
conversation with context.

### Run tests

```bash
pytest tests/
```

Tests mock the RAG chain, so they run without live API keys or network calls.

## Limitations & Roadmap

This is Phase 1 of the project: a working, context-aware RAG chatbot with a
clean structure to build on. The following are **explicitly not implemented
yet** and would be required before any real-world (non-demo) use:

- **No authentication or RBAC** &mdash; the API is unauthenticated. Anyone who can
  reach it can use it.
- **No audit logging or consent management** &mdash; required for any HIPAA/GDPR-relevant
  deployment.
- **No encryption-at-rest for conversation data** &mdash; conversation history is
  in-process memory only (see `SessionHistoryStore`), not written to disk, but
  also not durable, not shared across multiple server instances, and lost on
  restart.
- **No real HIPAA/GDPR compliance** &mdash; that requires signed Business Associate
  Agreements with every vendor touching PHI (OpenAI/Google/Pinecone), a formal
  risk assessment, breach-notification procedures, and legal review. None of
  that exists for this project, and code alone cannot create it.
- **No clinician-reviewed triage logic** &mdash; the system prompt tells the model to
  flag possible emergencies and defer to a physician, but this has not been
  validated by a medical professional and should not be trusted for real
  triage decisions.
- **Limited automated test coverage** &mdash; current tests are smoke tests (health
  check, request validation, response shape) with the chain mocked. Edge-case
  testing (contraindication phrasing, adversarial prompts, drug-interaction
  questions) is not yet covered.
- **Rate limiting / abuse protection** &mdash; not implemented.

## Tech stack

- **Backend:** FastAPI, Uvicorn
- **RAG:** LangChain (`create_history_aware_retriever`, `create_retrieval_chain`)
- **Vector store:** Pinecone (serverless)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **LLM:** OpenAI (`gpt-4o-mini`) or Google Gemini (`gemini-flash-latest`), selected via `LLM_PROVIDER`
- **Frontend:** vanilla HTML/CSS/JS (no build step)
