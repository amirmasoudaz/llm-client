import uvicorn

from src.config import get_settings


def main():
    """Run the Dana API server."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
