"""Worker registration and heartbeat management."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class DispatcherClient:
    """
    Handles worker registration and heartbeat with dispatcher.
    Runs in background tasks started by the worker on startup.
    """

    def __init__(self) -> None:
        self.dispatcher_url = os.getenv(
            "DISPATCHER_URL", "http://nexusai-dispatcher:8080"
        ).rstrip("/")
        self.worker_id = os.getenv("HOSTNAME", "unknown-worker")
        self.worker_port = int(os.getenv("WORKER_PORT", "8080"))
        self.heartbeat_interval = float(
            os.getenv("WORKER_HEARTBEAT_INTERVAL", "5")
        )
        self.register_timeout = float(os.getenv("WORKER_REGISTER_TIMEOUT", "10"))
        self.heartbeat_timeout = float(os.getenv("WORKER_HEARTBEAT_TIMEOUT", "3"))

        # Determine worker address for dispatcher to call back
        # In K8s: use Pod DNS (worker-id.service.namespace.svc.cluster.local)
        # In Docker Compose: use container name
        worker_host = os.getenv("WORKER_HOST")
        if not worker_host:
            # Auto-detect: try K8s StatefulSet pattern first
            service_name = os.getenv("WORKER_SERVICE_NAME", "nexusai-worker")
            namespace = os.getenv("WORKER_NAMESPACE", "default")
            if "." in self.worker_id or "-" in self.worker_id:
                # Looks like a K8s Pod name or container name
                worker_host = f"{self.worker_id}.{service_name}.{namespace}.svc.cluster.local"
            else:
                worker_host = self.worker_id

        self.worker_addr = f"http://{worker_host}:{self.worker_port}"

        self._session: Optional[aiohttp.ClientSession] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._registered = asyncio.Event()

        # Stats for heartbeat reporting
        self.total_processed = 0
        self.total_errors = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy-create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.heartbeat_timeout)
            )
        return self._session

    async def register(self) -> bool:
        """
        Register this worker with the dispatcher.
        Returns True on success, False on failure.
        Called once on worker startup.
        """
        session = await self._ensure_session()
        url = f"{self.dispatcher_url}/register"
        payload = {"worker_id": self.worker_id, "addr": self.worker_addr}

        try:
            async with session.post(
                url, json=payload, timeout=self.register_timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(
                        "Worker registered with dispatcher worker_id=%s addr=%s",
                        self.worker_id,
                        self.worker_addr,
                    )
                    self._registered.set()
                    return True
                else:
                    text = await resp.text()
                    logger.error(
                        "Registration failed status=%s response=%s",
                        resp.status,
                        text[:200],
                    )
                    return False
        except Exception as e:
            logger.error(
                "Registration request failed url=%s error=%s", url, e
            )
            return False

    async def _heartbeat_loop(self) -> None:
        """
        Background task that sends periodic heartbeats to dispatcher.
        Runs until shutdown or registration fails.
        """
        # Wait for initial registration
        try:
            await asyncio.wait_for(self._registered.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("Registration did not complete within 30s, heartbeat aborted")
            return

        logger.info(
            "Worker heartbeat started interval=%ss", self.heartbeat_interval
        )

        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)

                session = await self._ensure_session()
                url = f"{self.dispatcher_url}/heartbeat"
                payload = {
                    "worker_id": self.worker_id,
                    "stats": {
                        "processed": self.total_processed,
                        "errors": self.total_errors,
                    },
                }

                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        # Heartbeat accepted
                        pass
                    elif resp.status == 404:
                        # Not registered - try to re-register
                        logger.warning(
                            "Heartbeat rejected (not registered), re-registering"
                        )
                        self._registered.clear()
                        success = await self.register()
                        if not success:
                            logger.error("Re-registration failed, heartbeat stopped")
                            break
                    else:
                        text = await resp.text()
                        logger.warning(
                            "Heartbeat failed status=%s response=%s",
                            resp.status,
                            text[:200],
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
                # Continue trying

    async def deregister(self) -> None:
        """
        Explicitly deregister from dispatcher during graceful shutdown.
        Best-effort only - dispatcher will auto-remove on heartbeat timeout anyway.
        """
        if not self._registered.is_set():
            return

        try:
            session = await self._ensure_session()
            url = f"{self.dispatcher_url}/deregister"
            payload = {"worker_id": self.worker_id}

            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("Worker deregistered worker_id=%s", self.worker_id)
                else:
                    logger.warning(
                        "Deregister response status=%s", resp.status
                    )
        except Exception as e:
            logger.warning("Deregister request failed: %s", e)

    async def start(self) -> None:
        """
        Start registration and heartbeat background tasks.
        Called by worker_main on startup.
        """
        # Initial registration with retries (blocking)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            success = await self.register()
            if success:
                break
            if attempt < max_retries:
                backoff = min(2.0 ** (attempt - 1), 30.0)
                logger.warning(
                    "Initial registration failed, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt,
                    max_retries,
                )
                await asyncio.sleep(backoff)
            else:
                logger.error(
                    "Initial registration failed after %d attempts, heartbeat loop will retry",
                    max_retries,
                )

        # Start background heartbeat
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """
        Stop heartbeat and deregister.
        Called by worker_main on shutdown.
        """
        self._shutdown.set()

        # Cancel heartbeat task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Deregister
        await self.deregister()

        # Close session
        if self._session is not None:
            await self._session.close()
            self._session = None
