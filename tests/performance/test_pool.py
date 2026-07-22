#!/usr/bin/env python3
"""
测试连接池的 borrow/release 是否导致串行化
"""
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.utils.processing import preprocess

TRITON_URL = os.getenv("TRITON_SERVER", "10.120.0.66:40008")


class SimplePool:
    """简化版连接池"""
    def __init__(self, url, size):
        self._pool = Queue(maxsize=size)
        for _ in range(size):
            client = grpcclient.InferenceServerClient(
                url=url,
                channel_args=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ]
            )
            self._pool.put(client)

    def acquire(self, timeout=10.0):
        return self._pool.get(timeout=timeout)

    def release(self, client):
        self._pool.put_nowait(client)


def build_input(img):
    input_shape = [640, 640]
    buf = preprocess(img, input_shape)
    buf = np.expand_dims(buf, axis=0).astype(np.float32)
    inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
    inputs[0].set_data_from_numpy(buf)
    outputs = [grpcclient.InferRequestedOutput("output")]
    return inputs, outputs


def infer_with_pool(pool, img, task_id):
    """使用连接池"""
    t0 = time.perf_counter()
    try:
        client = pool.acquire()
        try:
            inputs, outputs = build_input(img)
            client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs,
                        model_version="1", client_timeout=30000)
        finally:
            pool.release(client)
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": True}
    except Exception as e:
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": False, "error": str(e)}


def infer_independent(img, task_id):
    """独立客户端（不用池）"""
    t0 = time.perf_counter()
    try:
        client = grpcclient.InferenceServerClient(
            url=TRITON_URL,
            channel_args=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ]
        )
        inputs, outputs = build_input(img)
        client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs,
                    model_version="1", client_timeout=30000)
        client.close()
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": True}
    except Exception as e:
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图片: {args.image}")
        sys.exit(1)
    print(f"图片: {img.shape}, Triton: {TRITON_URL}, 并发: {args.concurrent}\n")

    # 测试 1：使用连接池（70个连接）
    print(f"{'='*80}")
    print(f"测试 1: 使用连接池（70个连接）")
    print(f"{'='*80}")
    pool = SimplePool(TRITON_URL, 70)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = [ex.submit(infer_with_pool, pool, img, i) for i in range(args.concurrent)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start
    durations = [r["duration"] for r in results if r["success"]]
    print(f"总耗时: {wall:.3f}s  QPS: {args.concurrent/wall:.2f}")
    print(f"平均单次: {sum(durations)/len(durations):.3f}s")
    print(f"并发倍数: {sum(durations)/wall:.2f}x  {'✅ 并发' if sum(durations)/wall > 3 else '❌ 串行'}")

    # 测试 2：独立客户端（不用池）
    print(f"\n{'='*80}")
    print(f"测试 2: 独立客户端（每次创建新client）")
    print(f"{'='*80}")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = [ex.submit(infer_independent, img, i) for i in range(args.concurrent)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start
    durations = [r["duration"] for r in results if r["success"]]
    print(f"总耗时: {wall:.3f}s  QPS: {args.concurrent/wall:.2f}")
    print(f"平均单次: {sum(durations)/len(durations):.3f}s")
    print(f"并发倍数: {sum(durations)/wall:.2f}x  {'✅ 并发' if sum(durations)/wall > 3 else '❌ 串行'}")
