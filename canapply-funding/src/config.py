import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # API Settings
    HOST_API: str = Field(default="127.0.0.1")
    PORT_API: int = Field(default=4003)
    DEBUG_MODE: bool = Field(default=False)

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None)

    # Pusher
    PUSHER_AUTH_KEY: Optional[str] = Field(default=None)
    PUSHER_AUTH_SECRET: Optional[str] = Field(default=None)
    PUSHER_AUTH_VERSION: Optional[str] = Field(default=None)
    PUSHER_APP_CLUSTER: Optional[str] = Field(default=None)
    PUSHER_APP_ID: Optional[str] = Field(default=None)

    # Recommender Logic
    ENABLE_SEMANTIC: bool = Field(default=True)
    DEFAULT_TOP_K: int = Field(default=10)
    RRF_K: int = Field(default=60)
    FUZZY_THRESHOLD: int = Field(default=75)
    MAX_CANDIDATES_PER_LIST: int = Field(default=60)
    TAG_MATCH_TOPK: int = Field(default=5)
    RELATED_TOPK: int = Field(default=5)
    NEIGHBOR_TOPK: int = Field(default=15)
    MAX_NEIGHBOR_GRAPH_ROWS: int = Field(default=12000)
    W_TAG_MATCH: float = Field(default=1.0)
    W_RELATED: float = Field(default=0.3)
    SHORT_QUERY_LEN: int = Field(default=2)
    FUZZY_SHORT_THRESHOLD: int = Field(default=60)
    RECOMMENDER_EMBED_BACKEND: str = Field(default="local")
    RECOMMENDER_LOCAL_EMBED_MODEL: str = Field(default="all-MiniLM-L6-v2")
    FUNDING_TAG_COLLECTION: str = Field(default="funding_tags")
    FUNDING_SUBFIELD_COLLECTION: str = Field(default="funding_subfields")
    TAGGER_MAX_CONCURRENCY: int = Field(default=64)
    TAG_EXPORT_TOPK: int = Field(default=20)

    # Reminders
    REMINDERS_ON: bool = Field(default=False)
    CHECK_INTERVAL_MINUTES: Optional[int] = Field(default=None)

    # Database (Default/Target)
    DB_HOST: str = Field(default="127.0.0.1")
    DB_PORT: int = Field(default=3306)
    DB_USER: str = Field(default="funding")
    DB_PASS: str = Field(default="secret")
    DB_NAME: str = Field(default="emaildb")
    DB_MIN: int = Field(default=1)
    DB_MAX: int = Field(default=10)

    # Migration Source DB
    SRC_DB_HOST: Optional[str] = Field(default=None)
    SRC_DB_PORT: int = Field(default=3306)
    SRC_DB_USER: Optional[str] = Field(default=None)
    SRC_DB_PASS: Optional[str] = Field(default=None)
    SRC_DB_NAME: Optional[str] = Field(default=None)

    # Migration Destination DB
    DEST_DB_HOST: Optional[str] = Field(default=None)
    DEST_DB_PORT: int = Field(default=3306)
    DEST_DB_USER: Optional[str] = Field(default=None)
    DEST_DB_PASS: Optional[str] = Field(default=None)
    DEST_DB_NAME: Optional[str] = Field(default=None)

    # FTP Settings
    FTP_HOST: str = Field(default="ftp.canapply.ca")
    FTP_PORT: int = Field(default=21)
    FTP_USERNAME: Optional[str] = Field(default=None)
    FTP_PASSWORD: Optional[str] = Field(default=None)
    FTP_ROOT: str = Field(default="platform/")

    # Cloning
    CLONE_CHUNK_SIZE: int = Field(default=2000)

    # Outreach Config
    RESEND_ALLOWED: bool = Field(default=False)

    class Config:
        case_sensitive = True

    @classmethod
    def load(cls):
        # Manual boolean conversion for environment variables as Pydantic BaseModel 
        # doesn't automatically handle "0"/"1" or "true"/"false" strings from os.environ
        # when we are not using Pydantic Settings.
        
        env_vars = {}
        for field_name, field in cls.model_fields.items():
            val = os.getenv(field_name)
            if val is not None:
                if field.annotation is bool:
                    env_vars[field_name] = val.lower() in ("true", "1", "yes")
                elif field.annotation is int:
                    env_vars[field_name] = int(val)
                elif field.annotation is float:
                    env_vars[field_name] = float(val)
                else:
                    env_vars[field_name] = val
            elif field.default is not None:
                # Keep default if not provided in env
                pass
        
        return cls(**env_vars)

# Instance to be used across the app
settings = Settings.load()
