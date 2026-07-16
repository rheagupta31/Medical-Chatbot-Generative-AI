import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("medicalbot", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field("aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field("us-east-1", alias="PINECONE_REGION")

    llm_provider: str = Field("openai", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-flash-lite-latest", alias="GEMINI_MODEL")

    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(384, alias="EMBEDDING_DIMENSION")
    top_k: int = Field(3, alias="TOP_K")
    max_history_turns: int = Field(6, alias="MAX_HISTORY_TURNS")
    data_dir: str = Field("Data/", alias="DATA_DIR")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # langchain-pinecone's from_texts/from_existing_index only read the API key
    # from this env var, not from a kwarg -- keep it in sync with our config.
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    return settings
