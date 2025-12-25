"""Logging utility module."""

import logging
import sys
from typing import Optional


class Logger:
    """Logger wrapper for the application."""
    
    _instance: Optional['Logger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def setup(self, debug: bool = False, quiet: bool = False) -> None:
        """
        Setup the logger.
        
        Args:
            debug: Enable debug logging
            quiet: Suppress all output except errors
        """
        self._logger = logging.getLogger('ask_llm')
        self._logger.handlers.clear()
        
        # Set log level
        if quiet:
            level = logging.ERROR
        elif debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        self._logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        
        # Create formatter
        if debug:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        if self._logger:
            self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        if self._logger:
            self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if self._logger:
            self._logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        if self._logger:
            self._logger.error(message)


# Global logger instance
logger = Logger()

