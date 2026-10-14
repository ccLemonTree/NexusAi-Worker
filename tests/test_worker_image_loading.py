import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np


def load_image_function(executor, decoder, download):
    path = Path(__file__).parents[1] / "inference" / "handler.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "load_image")
    scope = {
        "asyncio": asyncio, "os": os, "np": np, "Optional": __import__("typing").Optional,
        "cv2": SimpleNamespace(imdecode=decoder, IMREAD_COLOR=1),
        "io_executor": executor, "_download_eos_async": download,
        "logger": logging.getLogger("test_image_loading"),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["load_image"]


class ImageLoadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_eos_decode_runs_off_loop_without_stalling_other_coroutines(self):
        main_thread = threading.get_ident()
        decoded_on = []
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        async def download(_path):
            return b"encoded image"

        def decode(data, flags):
            decoded_on.append(threading.get_ident())
            self.assertEqual(data.tobytes(), b"encoded image")
            time.sleep(0.08)
            return image

        with ThreadPoolExecutor(max_workers=1) as executor:
            load = load_image_function(executor, decode, download)
            ticks = []

            async def heartbeat():
                for _ in range(4):
                    await asyncio.sleep(0.01)
                    ticks.append(time.monotonic())

            started = time.monotonic()
            result, _ = await asyncio.gather(load("sample.jpeg", True), heartbeat())

        self.assertIs(result, image)
        self.assertNotEqual(decoded_on, [main_thread], "decode must not block the event loop")
        self.assertLess(ticks[0] - started, 0.06)

    async def test_invalid_image_still_returns_none(self):
        async def download(_path):
            return b"invalid"
        with ThreadPoolExecutor(max_workers=1) as executor:
            load = load_image_function(executor, lambda *args: None, download)
            with self.assertLogs("test_image_loading", level="ERROR"):
                self.assertIsNone(await load("invalid.jpeg", True))

    async def test_download_timeout_still_returns_none(self):
        async def download(_path):
            raise asyncio.TimeoutError()
        with ThreadPoolExecutor(max_workers=1) as executor:
            load = load_image_function(executor, lambda *args: self.fail("must not decode"), download)
            with self.assertLogs("test_image_loading", level="ERROR"):
                self.assertIsNone(await load("slow.jpeg", True))


if __name__ == "__main__":
    unittest.main()
