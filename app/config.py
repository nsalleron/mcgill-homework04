"""Application configuration loaded from environment variables and .env."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Agentic RAG Knowledge Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI LLM configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model_name: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_NAME")

    # Langfuse observability configuration
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # Local vector store & RAG configuration
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="knowledge_base")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=4, alias="TOP_K", ge=1)
    max_distance_threshold: float = Field(
        default=1.5, alias="MAX_DISTANCE_THRESHOLD", gt=0.0
    )

    @property
    def langfuse_configured(self) -> bool:
        """Return True when Langfuse is enabled and both keys are present."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key.strip()
            and self.langfuse_secret_key.strip()
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
