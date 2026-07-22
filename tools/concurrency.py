import os
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from queue import Queue, Empty

import tritonclient.grpc as grpcclient
from dotenv import load_dotenv

_env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(_env_path, override=True)

DEFAULT_SCALE = 5
DEFAULT_HARD_LIMIT = 64


def _resolve_workers(env_key: str, default_scale: int, hard_cap: int) -> int:
    cpu_count = os.cpu_count() or 4
    calculated_default = min(hard_cap, cpu_count * default_scale)
    env_value = os.getenv(env_key)
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return calculated_default


@lru_cache(maxsize=None)
def get_executor(env_key: str, *, default_scale: int = DEFAULT_SCALE, hard_cap: int = DEFAULT_HARD_LIMIT,
                 prefix: str = "nexus") -> ThreadPoolExecutor:
    workers = _resolve_workers(env_key, default_scale, hard_cap)
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=prefix)


def get_inference_executor() -> ThreadPoolExecutor:
    return get_executor("ANALYSE_MAX_WORKERS", prefix="analyse")


def get_logic_executor() -> ThreadPoolExecutor:
    return get_executor("LOGIC_MAX_WORKERS", default_scale=3, prefix="logic")


def get_io_executor() -> ThreadPoolExecutor:
    """专用于 I/O 密集型任务（EOS下载、本地文件读取），避免和推理线程竞争。"""
    return get_executor("IO_MAX_WORKERS", default_scale=2, hard_cap=32, prefix="io")


class TritonClientPool:
    """
    gRPC 连接池。每个 worker 进程独立持有，避免多线程共用单连接的竞争。

    使用 headless service + round_robin 负载均衡策略：
    - TRITON_SERVER 指向普通 service（兜底，不改环境变量）
    - TRITON_SERVER_HEADLESS 指向 headless service（优先使用）
    - round_robin 让 gRPC 在请求级别均匀分发到所有 triton pod
    池大小由环境变量 TRITON_POOL_SIZE 控制，默认 16。
    """

    def __init__(self, url: str, pool_size: int = None):
        if pool_size is None:
            pool_size = int(os.getenv("TRITON_POOL_SIZE", "70"))

        # 优先使用 headless service 地址，没有则用传入的 url
        headless_url = os.getenv(
            "TRITON_SERVER_HEADLESS",
            url.replace(
                "triton-server.triton.svc.cluster.local",
                "triton-server-headless.triton.svc.cluster.local"
            )
        )

        self._pool: Queue = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            client = grpcclient.InferenceServerClient(
                url=headless_url,
                channel_args=[
                    # 关键：请求级别的轮询，而不是连接级别
                    ("grpc.lb_policy_name", "round_robin"),
                    # headless service 返回多个 IP，需要解析所有地址
                    ("grpc.service_config", '{"loadBalancingPolicy": "round_robin"}'),
                ]
            )
            self._pool.put(client)

    def acquire(self, timeout: float = 10.0):
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            logging.getLogger(__name__).warning(
                f"Triton 连接池已耗尽（pool_size={self._pool.maxsize}），"
                f"请增大 TRITON_POOL_SIZE，本次请求跳过推理返回空结果"
            )
            return None

    def release(self, client: grpcclient.InferenceServerClient) -> None:
        self._pool.put_nowait(client)

    class _Ctx:
        def __init__(self, pool: "TritonClientPool"):
            self._pool = pool
            self._client = None

        def __enter__(self):
            self._client = self._pool.acquire()
            return self._client  # 可能是 None

        def __exit__(self, *_):
            if self._client is not None:
                self._pool.release(self._client)
            # client 是 None 时不归还，直接跳过

    def borrow(self) -> "_Ctx":
        """with pool.borrow() as client: ..."""
        return self._Ctx(self)


_triton_pool: TritonClientPool | None = None
_triton_pool_vlm: TritonClientPool | None = None


def get_triton_pool() -> TritonClientPool:
    """返回进程级单例连接池，首次调用时按 TRITON_SERVER 环境变量初始化。"""
    global _triton_pool
    if _triton_pool is None:
        url = os.getenv("TRITON_SERVER", "localhost:8001")
        _triton_pool = TritonClientPool(url=url)
    return _triton_pool


def get_triton_pool_vlm() -> TritonClientPool:
    """大模型（fastvlm 等）专用连接池，指向独立的 Triton VLM 部署。
    若未配置 TRITON_SERVER_VLM，退回到 TRITON_SERVER（向后兼容）。"""
    global _triton_pool_vlm
    if _triton_pool_vlm is None:
        url = os.getenv("TRITON_SERVER_VLM") or os.getenv("TRITON_SERVER", "localhost:8001")
        _triton_pool_vlm = TritonClientPool(url=url)
    return _triton_pool_vlm
