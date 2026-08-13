from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiohttp import web

from tools.logger_tools import Inference_logger as logger

ProcessHandler = Callable[[bytes], Awaitable[dict]]


@dataclass
class WorkerState:
    accepting: bool = True
    in_flight: int = 0
    drained: asyncio.Event = field(default_factory=asyncio.Event)


STATE = web.AppKey("state", WorkerState)


def begin_drain(app: web.Application) -> None:
    state = app[STATE]
    state.accepting = False
    if state.in_flight == 0:
        state.drained.set()


async def wait_for_drain(app: web.Application, timeout: float) -> None:
    await asyncio.wait_for(app[STATE].drained.wait(), timeout=timeout)


def create_app(process_handler: ProcessHandler | None = None) -> web.Application:
    if process_handler is None:
        from inference.handler import process_message

        process_handler = process_message

    max_concurrent = max(1, int(os.getenv("WORKER_MAX_CONCURRENT", "16")))
    semaphore = asyncio.Semaphore(max_concurrent)
    token = os.environ["WORKER_AUTH_TOKEN"]
    if not token:
        raise ValueError("WORKER_AUTH_TOKEN must not be empty")

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def ready(_request: web.Request) -> web.Response:
        if not app[STATE].accepting:
            raise web.HTTPServiceUnavailable(
                text="worker is draining", headers={"X-Worker-State": "draining"}
            )
        return web.json_response({"status": "ready"})

    async def process(request: web.Request) -> web.Response:
        task_id = request.headers.get("X-Task-ID", "")
        if not task_id:
            raise web.HTTPBadRequest(text="X-Task-ID is required")
        if request.headers.get("Authorization") != f"Bearer {token}":
            raise web.HTTPUnauthorized(text="invalid worker token")
        raw = await request.read()
        if not raw:
            raise web.HTTPBadRequest(text="empty task payload")
        if not app[STATE].accepting:
            raise web.HTTPServiceUnavailable(
                text="worker is draining", headers={"X-Worker-State": "draining"}
            )
        if semaphore.locked():
            raise web.HTTPTooManyRequests(text="worker is at capacity")

        async with semaphore:
            state = app[STATE]
            state.in_flight += 1
            state.drained.clear()
            logger.info(f"Inference processing task_id={task_id}")
            try:
                result = await process_handler(raw)
            finally:
                state.in_flight -= 1
                if state.in_flight == 0:
                    state.drained.set()
        return web.json_response(result)

    app = web.Application(
        client_max_size=int(os.getenv("WORKER_MAX_REQUEST_BYTES", str(1024 * 1024)))
    )
    app[STATE] = WorkerState()
    app[STATE].drained.set()
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    app.router.add_post("/process", process)
    return app
