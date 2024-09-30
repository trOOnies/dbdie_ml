"""Settings for the project."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    dbdie_main_fd: str
    fastapi_host: str
    base_api_host: str
    check_rps: str

    class Config:
        env_file = ".env"


ST = Settings()
