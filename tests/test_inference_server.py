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

        environment = patch.dict(os.environ, {"WORKER_AUTH_TOKEN": "test-token"})
        environment.start()
        self.addCleanup(environment.stop)

        self.app = create_app(process_handler=process_message)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()

    async def test_drain_marks_worker_unready_and_rejects_new_work(self):
        response = await self.client.get("/ready")
        self.assertEqual(200, response.status)

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


if __name__ == "__main__":
    unittest.main()
