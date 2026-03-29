import asyncio


class ResponseTimeoutError(asyncio.TimeoutError):
    """Raised when get_response exceeds the user-supplied timeout."""


__all__ = ["ResponseTimeoutError"]
