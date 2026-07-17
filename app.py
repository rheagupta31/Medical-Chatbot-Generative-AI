import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag_chain import MedicalRagChain

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
# keep our INFO logs (timings, errors) but silence noisy library chatter
for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("medibot")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Walk the exception chain looking for a provider rate-limit (HTTP 429)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower():
            return True
        current = current.__cause__ or current.__context__
    return False

templates = Jinja2Templates(directory="templates")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _state["chain"] = MedicalRagChain(settings)
    yield
    _state.clear()


app = FastAPI(title="Medical Chatbot (Portfolio Demo)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    disclaimer: str
    sources: list[str]
    session_id: str


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    chain: MedicalRagChain | None = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready yet")

    session_id = payload.session_id or str(uuid.uuid4())
    try:
        result = chain.ask(session_id, payload.message)
    except Exception as exc:
        logger.exception("chat request failed (session %s)", session_id)
        if _is_rate_limit_error(exc):
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI provider's rate limit was reached. This may be the "
                    "free-tier daily quota, which resets at midnight Pacific time."
                ),
            ) from exc
        # generic 500 -- details stay server-side
        raise HTTPException(status_code=500, detail="Failed to generate an answer") from exc

    return ChatResponse(session_id=session_id, **result)
