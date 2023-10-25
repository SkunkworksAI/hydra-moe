"""
This module manages the settings for the application, such as API version, CORS origins.
"""
from loguru import logger
import logging
import pathlib
import sys

from dotenv import load_dotenv, find_dotenv
from pydantic import AnyHttpUrl, EmailStr, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Union

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Defines the project root directory
ROOT = pathlib.Path(__file__).resolve().parent.parent


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are ints

class Settings(BaseSettings):
    """
    Settings model that parses and validates the environment variables.
    """

    # API version
    API_V1_STR: str = "/api/v1"

    # List of origins for CORS (Cross-Origin Resource Sharing)
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:8000"]

    # Origins that match this regex OR are in the above list are allowed
    BACKEND_CORS_ORIGIN_REGEX: Optional[str] = ""

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        A validator for the CORS origins to ensure that they are in the correct format.

        Args:
            v (Union[str, List[str]]): The CORS origins, either as a string or a list of strings.

        Returns:
            Union[List[str], str]: The validated CORS origins.
        """
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError(v)

    logging: LoggingSettings = LoggingSettings()

    class Config:
        case_sensitive = True

# Instantiate the settings
settings = Settings()