# src/tools/logger.py

import logging.config
import os
from typing import Tuple
from pathlib import Path

from dotenv import load_dotenv, find_dotenv


class Logger:
    load_dotenv(find_dotenv(".env"))

    # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOGGING_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")
    MAX_BYTES = 1 * 1024 * 1024  # 1 MB
    BACKUP_COUNT = 1024  # 1 GB

    def __init__(self):
        self._cpd = Path(__file__).resolve().parent.parent.parent
        self._path = self._cpd / "logs"
        self._path.mkdir(parents=True, exist_ok=True)

    def create(
            self,
            application: str,
            file_name: str = None,
            logger_name: str = None,
            logging_level: str = None,
            config_only: bool = False
    ) -> Tuple[logging.Logger or None, dict]:
        config_dict = self._config(
            file_path=self._path / f"{file_name if file_name else application}.log",
            app_names=["uvicorn", "fastapi_app"] if application == "api" else [application],
            logging_level=logging_level if logging_level else self.LOGGING_LEVEL
        )

        if config_only:
            return None, config_dict

        logging.config.dictConfig(config_dict)
        logger = logging.getLogger(logger_name if logger_name else application)

        return logger, config_dict

    def _config(self, file_path, app_names, logging_level):
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging_level,
                    "maxBytes": self.MAX_BYTES,
                    "backupCount": self.BACKUP_COUNT,
                    "filename": file_path,
                    "formatter": "default",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                app: {
                    "handlers": ["file", "console"],
                    "level": logging_level,
                } for app in app_names
            },
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            }
        }
