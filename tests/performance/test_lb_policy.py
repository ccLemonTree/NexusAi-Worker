#!/usr/bin/env python3
"""
验证 round_robin 负载均衡策略是否导致串行化
对比两种 gRPC 客户端配置的并发性能
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


def infer_once(client, img, task_id):
    inputs, outputs = build_input(img)
    t0 = time.perf_counter()
    try:
        client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs,
                     model_version="1", client_timeout=30000)
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": True}
    except Exception as e:
        return {"task_id": task_id, "duration": time.perf_counter() - t0, "success": False, "error": str(e)}


def run_test(label, channel_args, concurrent, img, shared=False):
    print(f"\n{'='*80}")
    print(f"配置: {label}")
    print(f"{'='*80}")

    if shared:
        # 模拟连接池：共享少量客户端
        clients = [grpcclient.InferenceServerClient(url=TRITON_URL, channel_args=channel_args)
                   for _ in range(concurrent)]
    else:
        clients = [grpcclient.InferenceServerClient(url=TRITON_URL, channel_args=channel_args)
                   for _ in range(concurrent)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        futures = [ex.submit(infer_once, clients[i], img, i) for i in range(concurrent)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start

    for c in clients:
        c.close()

    durations = [r["duration"] for r in results if r["success"]]
    total = sum(durations)
    print(f"总耗时: {wall:.3f}s  QPS: {concurrent/wall:.2f}")
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
    print(f"图片: {img.shape}, Triton: {TRITON_URL}")

    # 配置 A：只有 max message length（和 test_raw_grpc 一样，快）
    run_test(
        "A. 无 round_robin（仅 max message length）",
        [
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
        args.concurrent, img
    )

    # 配置 B：带 round_robin（和连接池一样，怀疑是慢的元凶）
    run_test(
        "B. 带 round_robin lb_policy（连接池配置）",
        [
            ("grpc.lb_policy_name", "round_robin"),
            ("grpc.service_config", '{"loadBalancingPolicy": "round_robin"}'),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
        args.concurrent, img
    )
