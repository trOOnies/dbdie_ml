"""Settings for the project."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    fastapi_host: str
    dbdie_main_fd: str

    class Config:
        env_file = ".env"


ST = Settings()
