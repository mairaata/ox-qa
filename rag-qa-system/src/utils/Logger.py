import logging
from functools import lru_cache

class Logger:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._logger = cls._instance._initialize_logger()
        return cls._instance

    @staticmethod
    @lru_cache(maxsize=None)
    def _initialize_logger():
        logger = logging.getLogger('RAG')
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @staticmethod
    def get_logger():
        return Logger()._logger