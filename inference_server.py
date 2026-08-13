from __future__ import annotations

import asyncio
import os

from aiohttp import web

from inference.handler import process_message
from tools.logger_tools import Inference_logger as logger


def create_app() -> web.Application:
    max_concurrent = max(1, int(os.getenv("WORKER_MAX_CONCURRENT", "16")))
    semaphore = asyncio.Semaphore(max_concurrent)
    token = os.environ["WORKER_AUTH_TOKEN"]
    if not token:
        raise ValueError("WORKER_AUTH_TOKEN must not be empty")

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def process(request: web.Request) -> web.Response:
        task_id = request.headers.get("X-Task-ID", "")
        if not task_id:
            raise web.HTTPBadRequest(text="X-Task-ID is required")
        if request.headers.get("Authorization") != f"Bearer {token}":
            raise web.HTTPUnauthorized(text="invalid worker token")
        if semaphore.locked():
            raise web.HTTPServiceUnavailable(text="worker is at capacity")

        raw = await request.read()
        if not raw:
            raise web.HTTPBadRequest(text="empty task payload")

        async with semaphore:
            logger.info(f"Inference processing task_id={task_id}")
            result = await process_message(raw)
        return web.json_response(result)

    app = web.Application(
        client_max_size=int(os.getenv("WORKER_MAX_REQUEST_BYTES", str(1024 * 1024)))
    )
    app.router.add_get("/health", health)
    app.router.add_post("/process", process)
    return app
