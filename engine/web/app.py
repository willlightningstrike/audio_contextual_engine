"""FastAPI composition for the local Contextual Engine API."""

from __future__ import annotations

from urllib.parse import urlsplit
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine.web.routes import build_router


def create_app(*, store: object, runner: object, settings: object) -> FastAPI:
    app = FastAPI(title="Contextual Engine", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.runner = runner
    app.state.settings = settings
    app.include_router(build_router(store=store, runner=runner, settings=settings))
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.middleware("http")
    async def reject_unexpected_origins(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    host = (parsed.hostname or "").lower()
                    expected_authority = request.headers.get("host", "").lower()
                    if (
                        parsed.scheme != "http"
                        or host not in {"localhost", "127.0.0.1", "::1"}
                        or parsed.netloc.lower() != expected_authority
                    ):
                        raise ValueError
                except ValueError:
                    return JSONResponse(
                        {"detail": "unexpected browser origin"}, status_code=403
                    )
        return await call_next(request)

    return app
