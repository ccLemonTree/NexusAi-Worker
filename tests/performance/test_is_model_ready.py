#!/usr/bin/env python3
"""
测试 is_model_ready() 是否导致串行化
"""
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.utils.processing import preprocess

TRITON_URL = os.getenv("TRITON_SERVER", "10.120.0.66:40008")


def build_input(img):
    input_shape = [640, 640]
    buf = preprocess(img, input_shape)
    buf = np.expand_dims(buf, axis=0).astype(np.float32)
    inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
    inputs[0].set_data_from_numpy(buf)
    outputs = [grpcclient.InferRequestedOutput("output")]
    return inputs, outputs


def infer_with_ready_check(client, img, task_id):
    """带 is_model_ready 检查"""
    inputs, outputs = build_input(img)
    t0 = time.perf_counter()
    try:
        # 模拟 yolov5_detector.py 的调用模式
        if client.is_model_ready("yolov5_persondog", "1"):
            client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs,
                        model_version="1", client_timeout=30000)
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": True}
    except Exception as e:
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": False, "error": str(e)}


def infer_without_ready_check(client, img, task_id):
    """不带 is_model_ready 检查"""
    inputs, outputs = build_input(img)
    t0 = time.perf_counter()
    try:
        client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs,
                     model_version="1", client_timeout=30000)
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": True}
    except Exception as e:
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": False, "error": str(e)}


def run_test(label, infer_func, concurrent, img):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")

    clients = [grpcclient.InferenceServerClient(
        url=TRITON_URL,
        channel_args=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    ) for _ in range(concurrent)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        futures = [ex.submit(infer_func, clients[i], img, i) for i in range(concurrent)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start

    for c in clients:
        c.close()

    success = [r for r in results if r["success"]]
    durations = [r["duration"] for r in success]
    total = sum(durations)

    print(f"总耗时: {wall:.3f}s  QPS: {concurrent/wall:.2f}")
    if durations:
        print(f"平均单次: {sum(durations)/len(durations):.3f}s")
        print(f"并发倍数: {total/wall:.2f}x  {'✅ 并发' if total/wall > 3 else '❌ 串行'}")


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
    print(f"图片: {img.shape}, Triton: {TRITON_URL}, 并发: {args.concurrent}")

    # 测试 1：不带 is_model_ready（基线，应该快）
    run_test("测试 1: 不带 is_model_ready() 检查", infer_without_ready_check, args.concurrent, img)

    # 测试 2：带 is_model_ready（怀疑慢）
    run_test("测试 2: 带 is_model_ready() 检查", infer_with_ready_check, args.concurrent, img)
