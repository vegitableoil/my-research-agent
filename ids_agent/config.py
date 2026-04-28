"""System-wide configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 8192

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-large"
    openai_api_key: str = ""  # used only for embeddings if selected

    # ── Vector DB ────────────────────────────────────────────────────────────
    vector_db_backend: str = "chroma"          # chroma | weaviate | pgvector
    chroma_persist_dir: Path = Path("data/chroma")
    weaviate_url: str = "http://localhost:8080"

    # ── Relational DB ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///data/ids_agent.db"

    # ── Artifact Store ───────────────────────────────────────────────────────
    artifact_root: Path = Path("data/artifacts")
    use_s3: bool = False
    s3_bucket: str = ""
    s3_endpoint: str = ""

    # ── Paper Sources ────────────────────────────────────────────────────────
    semantic_scholar_api_key: str = ""
    ieee_api_key: str = ""
    arxiv_max_results: int = 50

    # ── GitHub ───────────────────────────────────────────────────────────────
    github_token: str = ""

    # ── Agent Behaviour ──────────────────────────────────────────────────────
    max_verification_retries: int = 3
    max_replication_retries: int = 3
    hpo_n_trials: int = 100

    # ── Human Loop ───────────────────────────────────────────────────────────
    human_loop_backend: str = "cli"            # cli | webhook
    webhook_gate_url: str = "http://localhost:8888/review"
    notification_email: str = ""
    slack_webhook_url: str = ""


settings = Settings()
