from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
            timeout=30,
            max_retries=1,
        )

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
            temperature=0.2,
            timeout=30,
            max_retries=1,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
