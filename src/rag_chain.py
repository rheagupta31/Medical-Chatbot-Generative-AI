import logging
import threading
import time
from collections import defaultdict, deque

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore

from src.config import Settings
from src.helper import download_hugging_face_embeddings
from src.llm import get_llm
from src.prompt import prompt

logger = logging.getLogger("medibot")

DISCLAIMER = (
    "This information is for general educational purposes only and is NOT a substitute "
    "for professional medical advice, diagnosis, or treatment. Always seek the advice of "
    "a qualified physician or other health provider with any questions about a medical "
    "condition. If this is a medical emergency, call your local emergency number immediately."
)

_CONDENSE_SYSTEM_PROMPT = (
    "You rewrite user questions into search queries for a medical reference encyclopedia. "
    "Given the chat history and the latest user question, produce ONE standalone search "
    "query that can be understood without the chat history. Replace colloquial or informal "
    "terms with standard medical terminology and include common synonyms, e.g. "
    "'acidity' -> 'heartburn acid reflux GERD indigestion', "
    "'sugar' (as a condition) -> 'diabetes blood glucose', "
    "'BP' -> 'blood pressure hypertension'. "
    "Do not answer the question; output only the rewritten query."
)
_condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


class SessionHistoryStore:
    """In-memory per-session chat history.

    Known limitation (Phase 1): lost on process restart and not shared across
    multiple worker processes. Replace with a real store (e.g. Redis) before
    any multi-instance deployment.
    """

    def __init__(self, max_turns: int):
        self._max_turns = max_turns
        self._lock = threading.Lock()
        self._sessions: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_turns * 2)
        )

    def get(self, session_id: str) -> list:
        with self._lock:
            return list(self._sessions[session_id])

    def append(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            history = self._sessions[session_id]
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))


class MedicalRagChain:
    def __init__(self, settings: Settings):
        embeddings = download_hugging_face_embeddings(settings.embedding_model)
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
        )
        self._retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": settings.top_k}
        )
        llm = get_llm(settings)
        self._condense_chain = _condense_prompt | llm | StrOutputParser()
        self._answer_chain = prompt | llm | StrOutputParser()
        self._history = SessionHistoryStore(max_turns=settings.max_history_turns)

    def ask(self, session_id: str, question: str, return_context: bool = False) -> dict:
        chat_history = self._history.get(session_id)

        # Rewrite before retrieval: resolves follow-up references against the
        # chat history and maps colloquial phrasing to clinical terminology so
        # the vector search hits the right chunks. Falls back to the raw
        # question if the rewrite call fails.
        t0 = time.perf_counter()
        try:
            search_query = self._condense_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )
        except Exception:
            search_query = question
        t_rewrite = time.perf_counter()

        docs = self._retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs)
        t_retrieve = time.perf_counter()

        answer = self._answer_chain.invoke(
            {"input": question, "chat_history": chat_history, "context": context}
        )
        t_answer = time.perf_counter()
        logger.info(
            "timings rewrite=%.1fs retrieve=%.1fs answer=%.1fs total=%.1fs",
            t_rewrite - t0,
            t_retrieve - t_rewrite,
            t_answer - t_retrieve,
            t_answer - t0,
        )
        self._history.append(session_id, question, answer)

        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        result = {"answer": answer, "disclaimer": DISCLAIMER, "sources": sources}
        if return_context:
            result["context_text"] = context
        return result
