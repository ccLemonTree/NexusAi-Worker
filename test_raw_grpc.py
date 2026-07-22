#!/usr/bin/env python3
"""
最底层测试：直接使用 Triton gRPC 客户端
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


def direct_grpc_call(client, img, task_id):
    """直接调用 Triton gRPC"""
    # 预处理
    input_shape = [640, 640]
    input_image_buffer = preprocess(img, input_shape)
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    input_image = input_image_buffer.astype(np.float32)

    # 构造请求
    inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
    inputs[0].set_data_from_numpy(input_image)
    outputs = [grpcclient.InferRequestedOutput("output")]

    t0 = time.perf_counter()
    try:
        results = client.infer(
            model_name="yolov5_persondog",
            inputs=inputs,
            outputs=outputs,
            model_version="1",
            client_timeout=30000
        )
        t1 = time.perf_counter()
        output_data = results.as_numpy("output")
        return {"task_id": task_id, "duration": t1 - t0, "success": True, "shape": output_data.shape}
    except Exception as e:
        t1 = time.perf_counter()
        return {"task_id": task_id, "duration": t1 - t0, "success": False, "error": str(e)}


def test_raw_grpc(image_path: str, concurrent: int):
    """测试原始 gRPC 调用的并发性"""
    print(f"\n{'='*80}")
    print(f"原始 gRPC 并发测试")
    print(f"{'='*80}\n")

    img = cv2.imread(image_path)
    print(f"图片尺寸: {img.shape}\n")

    # 创建多个 gRPC 客户端
    clients = []
    for i in range(concurrent):
        client = grpcclient.InferenceServerClient(
            url="10.120.0.66:40008",
            channel_args=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ]
        )
        clients.append(client)

    print(f"创建了 {len(clients)} 个独立的 gRPC 客户端\n")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(direct_grpc_call, clients[i % len(clients)], img, i)
            for i in range(concurrent)
        ]

        results = [f.result() for f in futures]

    end_time = time.perf_counter()

    # 统计
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}")
    print(f"总耗时: {end_time - start_time:.3f} 秒")
    print(f"成功: {success_count}/{concurrent}")
    print(f"QPS: {concurrent / (end_time - start_time):.2f}")

    durations = [r["duration"] for r in results if r["success"]]
    if durations:
        print(f"\n推理耗时:")
        print(f"  最小: {min(durations):.3f}s")
        print(f"  最大: {max(durations):.3f}s")
        print(f"  平均: {sum(durations)/len(durations):.3f}s")

        total_inference_time = sum(durations)
        wall_time = end_time - start_time

        print(f"\n并发分析:")
        print(f"  所有推理时间总和: {total_inference_time:.3f}s")
        print(f"  实际墙上时间: {wall_time:.3f}s")
        print(f"  并发倍数: {total_inference_time / wall_time:.2f}x")

        if total_inference_time / wall_time > 10:
            print(f"  ✅ 真正并发！")
        elif total_inference_time / wall_time > 2:
            print(f"  ⚠️  部分并发")
        else:
            print(f"  ❌ 完全串行")

    print(f"{'='*80}\n")

    # 关闭客户端
    for client in clients:
        client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    test_raw_grpc(args.image, args.concurrent)
