from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # App
    app_env: str
    debug: bool
    job_poll_interval_s: float
    job_max_concurrency: int

    # DB
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    db_min_size: int
    db_max_size: int
    
    # Redis
    redis_url: str

    # Storage
    storage_backend: str  # "s3" or "local"
    local_storage_root: str
    s3_region: str
    s3_bucket: str

    # LLM
    llm_mode: str  # "live" or "mock"
    llm_review_model: str
    llm_revision_model: str
    llm_doc_model: str
    llm_orchestrator_model: str
    
    # Platform Integration
    platform_backend_url: str
    platform_webhook_url: str
    platform_api_key: str

    # Dev
    dev_student_id: int
    
    @property
    def database_url(self) -> str:
        """Generate database URL for Prisma."""
        return f"mysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    def _bool(name: str, default: str = "false") -> bool:
        return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}

    return Settings(
        # App
        app_env=os.getenv("APP_ENV", "dev"),
        debug=_bool("DEBUG", "true"),
        job_poll_interval_s=float(os.getenv("JOB_POLL_INTERVAL_S", "0.5")),
        job_max_concurrency=int(os.getenv("JOB_MAX_CONCURRENCY", "2")),
        
        # DB
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=int(os.getenv("DB_PORT", "3306")),
        db_user=os.getenv("DB_USER", "root"),
        db_password=os.getenv("DB_PASSWORD", ""),
        db_name=os.getenv("DB_NAME", "dana"),
        db_min_size=int(os.getenv("DB_POOL_MIN", "1")),
        db_max_size=int(os.getenv("DB_POOL_MAX", "10")),
        
        # Redis
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        
        # Storage
        storage_backend=os.getenv("STORAGE_BACKEND", "s3"),
        local_storage_root=os.getenv("LOCAL_STORAGE_ROOT", "./data"),
        s3_region=os.getenv("S3_REGION", "ca-central-1"),
        s3_bucket=os.getenv("S3_BUCKET", "canapply-platform-prod"),
        
        # LLM
        llm_mode=os.getenv("LLM_MODE", "live"),
        llm_review_model=os.getenv("LLM_REVIEW_MODEL", "gpt-4o-mini"),
        llm_revision_model=os.getenv("LLM_REVISION_MODEL", "gpt-4o-mini"),
        llm_doc_model=os.getenv("LLM_DOC_MODEL", "gpt-4o-mini"),
        llm_orchestrator_model=os.getenv("LLM_ORCHESTRATOR_MODEL", "gpt-4o"),
        
        # Platform Integration
        platform_backend_url=os.getenv("PLATFORM_BACKEND_URL", "http://localhost:8080"),
        platform_webhook_url=os.getenv("PLATFORM_WEBHOOK_URL", ""),
        platform_api_key=os.getenv("PLATFORM_API_KEY", ""),
        
        # Dev
        dev_student_id=int(os.getenv("DEV_STUDENT_ID", "0")),
    )
