import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

handler_module = types.ModuleType("inference.handler")


async def _unused_handler(_raw: bytes) -> dict:
    raise AssertionError("test must inject its own process handler")


handler_module.process_message = _unused_handler
with patch.dict(sys.modules, {"inference.handler": handler_module}):
    from inference_server import begin_drain, create_app


class InferenceServerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

        async def process_message(_raw: bytes) -> dict:
            self.started.set()
            await self.release.wait()
            return {"id": 1}

        environment = patch.dict(
            os.environ,
            {
                "WORKER_AUTH_TOKEN": "test-token",
                "WORKER_MAX_CONCURRENT": "1",
                "WORKER_CAPACITY_WEIGHT": "2.5",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

        with patch("socket.gethostname", return_value="nexusai-worker-test"):
            self.app = create_app(process_handler=process_message)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()

    async def test_drain_marks_worker_unready_and_rejects_new_work(self):
        response = await self.client.get("/ready")
        self.assertEqual(200, response.status)
        self.assertEqual("nexusai-worker-test", response.headers["X-Worker-ID"])
        self.assertEqual("2.5", response.headers["X-Worker-Weight"])
        self.assertEqual("1", response.headers["X-Worker-Max-Concurrent"])

        begin_drain(self.app)

        response = await self.client.get("/ready")
        self.assertEqual(503, response.status)
        response = await self.client.post(
            "/process",
            data=b"{}",
            headers={
                "Authorization": "Bearer test-token",
                "X-Task-ID": "topic:0:1",
            },
        )
        self.assertEqual(503, response.status)
        self.assertEqual("nexusai-worker-test", response.headers["X-Worker-ID"])
        self.assertEqual("2.5", response.headers["X-Worker-Weight"])
        self.assertEqual("1", response.headers["X-Worker-Max-Concurrent"])

    async def test_drain_allows_accepted_work_to_finish(self):
        request = asyncio.create_task(
            self.client.post(
                "/process",
                data=b"{}",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Task-ID": "topic:0:1",
                },
            )
        )
        await self.started.wait()

        begin_drain(self.app)
        self.release.set()

        response = await request
        self.assertEqual(200, response.status)
        self.assertEqual("nexusai-worker-test", response.headers["X-Worker-ID"])
        self.assertEqual("2.5", response.headers["X-Worker-Weight"])
        self.assertEqual("1", response.headers["X-Worker-Max-Concurrent"])

    async def test_capacity_rejection_includes_worker_capacity_headers(self):
        request = asyncio.create_task(
            self.client.post(
                "/process",
                data=b"{}",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Task-ID": "topic:0:1",
                },
            )
        )
        await self.started.wait()

        response = await self.client.post(
            "/process",
            data=b"{}",
            headers={
                "Authorization": "Bearer test-token",
                "X-Task-ID": "topic:0:2",
            },
        )
        self.assertEqual(429, response.status)
        self.assertEqual("2.5", response.headers["X-Worker-Weight"])
        self.assertEqual("1", response.headers["X-Worker-Max-Concurrent"])

        self.release.set()
        await request

    async def test_invalid_capacity_configuration_fails_startup(self):
        for name, value in (
            ("WORKER_CAPACITY_WEIGHT", "0"),
            ("WORKER_CAPACITY_WEIGHT", "nan"),
            ("WORKER_MAX_CONCURRENT", "0"),
        ):
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValueError):
                        create_app(process_handler=_unused_handler)


if __name__ == "__main__":
    unittest.main()
