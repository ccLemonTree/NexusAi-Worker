"""NexusAi inference service entrypoint."""

import asyncio
import os
import pathlib
import signal

from aiohttp import web
from dotenv import load_dotenv

from tools.logger_tools import Inference_logger as logger


async def run_server() -> None:
    from inference_server import begin_drain, create_app, wait_for_drain

    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8080"))
    propagation_delay = float(os.getenv("WORKER_DRAIN_DELAY", "5"))
    shutdown_timeout = float(os.getenv("WORKER_SHUTDOWN_TIMEOUT", "90"))

    app = create_app()
    runner = web.AppRunner(
        app,
        handle_signals=False,
        access_log=None,
        shutdown_timeout=1,
    )
    await runner.setup()
    await web.TCPSite(runner, host=host, port=port).start()

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stopping.set)
    except NotImplementedError:
        signal.signal(
            signal.SIGINT,
            lambda _sig, _frame: loop.call_soon_threadsafe(stopping.set),
        )

    logger.info(f"NexusAi inference service listening on {host}:{port}")
    try:
        await stopping.wait()
        begin_drain(app)
        logger.info("NexusAi inference service draining")
        await asyncio.sleep(propagation_delay)
        try:
            await wait_for_drain(app, shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("Inference drain timed out after %ss", shutdown_timeout)
    finally:
        await runner.cleanup()


def main() -> None:
    load_dotenv(pathlib.Path(__file__).parent / ".env", override=False)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
