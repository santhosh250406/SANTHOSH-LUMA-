# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache # <--- Import lru_cache here

class Settings(BaseSettings):
    # Load settings from the .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        frozen=True  # <--- !! ADD THIS LINE !!
    )

    # Azure OpenAI Credentials
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str

# Create a single, cached instance of the settings
# We will "depend" on this function in our API
@lru_cache  # <--- Add cache here for efficiency
def get_settings() -> Settings:
    return Settings()