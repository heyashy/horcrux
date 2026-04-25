"""Centralised, typed config for the Horcrux lab.

Single source of truth for every environment variable, secret, and tunable.
Imported as `from horcrux.config import settings` everywhere else; no
`os.getenv` calls in core logic.

Required env vars fail validation at import time — a missing
`ANTHROPIC_API_KEY` raises before any LLM call is made, not after a 7-minute
ingest run.

See ADR-0004 for the rationale.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6333
    paragraphs_collection: str = "hp_paragraphs"
    chapters_collection: str = "hp_chapters"


class TemporalConfig(BaseSettings):
    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "horcrux-ingest"


class LiteLLMConfig(BaseSettings):
    base_url: str = "http://localhost:4000"
    haiku_alias: str = "haiku"
    sonnet_alias: str = "sonnet"


class LangSmithConfig(BaseSettings):
    api_key: SecretStr | None = None
    project: str = "horcrux"
    tracing_enabled: bool = True


class EmbeddingConfig(BaseSettings):
    model_name: str = "BAAI/bge-large-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    dim: int = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: SecretStr
    corpus_path: str = "data_lake/corpus.pdf"
    checkpointer_db: str = "horcrux.db"

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    litellm: LiteLLMConfig = Field(default_factory=LiteLLMConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


settings = Settings()
