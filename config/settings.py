"""
ToxiRAG Configuration Settings
Centralized configuration using pydantic-settings for type safety and validation.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class ToxiRAGSettings(BaseSettings):
    """Main configuration class for ToxiRAG application."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Model Configuration
    openai_embed_model: str = "text-embedding-3-large"
    default_llm_provider: Literal["openai", "gemini"] = "openai"
    default_llm_model: str = "gpt-5-nano"
    gemini_model: str = "gemini-2.5-flash"
    
    # Vector Database
    # Production database location - moved from tmp to persistent storage
    lancedb_uri: str = "data/knowledge_base/lancedb"
    collection_name: str = "toxicology_docs"
    
    # Retrieval Configuration
    retrieval_top_k: int = 5
    retrieval_temperature: float = 0.1
    max_tokens: int = 2000
    
    # Data Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    units_version: str = "v1.0"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/toxirag.log"
    
    # Development
    debug: bool = False
    environment: str = "development"
    
    # Network Configuration
    no_proxy: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env
    
    @validator("lancedb_uri")
    def validate_lancedb_uri(cls, v):
        """Ensure LanceDB URI directory exists or can be created."""
        path = Path(v).parent
        path.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("log_file")
    def validate_log_file(cls, v):
        """Ensure log directory exists."""
        path = Path(v).parent
        path.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("no_proxy")
    def set_no_proxy_env(cls, v):
        """Set NO_PROXY environment variable for HTTP libraries."""
        if v:
            # Combine with existing NO_PROXY if present
            current_no_proxy = os.environ.get('NO_PROXY', '')
            if current_no_proxy and v not in current_no_proxy:
                new_no_proxy = f"{current_no_proxy},{v}"
            else:
                new_no_proxy = v
            os.environ['NO_PROXY'] = new_no_proxy
        return v
    
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key.strip()) > 0
    
    def has_google_key(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None and len(self.google_api_key.strip()) > 0
    
    def has_any_llm_key(self) -> bool:
        """Check if at least one LLM API key is configured."""
        return self.has_openai_key() or self.has_google_key()


# Global settings instance
settings = ToxiRAGSettings()
