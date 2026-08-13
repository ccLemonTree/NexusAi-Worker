"""NexusAi inference service entrypoint."""

import os
import pathlib

from aiohttp import web
from dotenv import load_dotenv

from tools.logger_tools import Inference_logger as logger


def main() -> None:
    load_dotenv(pathlib.Path(__file__).parent / ".env", override=False)
    from inference_server import create_app

    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8080"))
    logger.info(f"NexusAi inference service listening on {host}:{port}")
    web.run_app(create_app(), host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()
