#!/usr/bin/env python3
"""
测试 postprocess 是否导致串行
"""
import time
import cv2
import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.utils.processing import preprocess, postprocess

def inference_with_postprocess(img, triton_url):
    """推理 + 后处理（完全模拟 yolov5）"""
    start = time.time()

    client = grpcclient.InferenceServerClient(
        url=triton_url,
        channel_args=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    )

    try:
        # 预处理
        input_shape = [640, 640]
        input_image_buffer = preprocess(img, input_shape)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        input_image = input_image_buffer.astype(np.float32)

        inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [grpcclient.InferRequestedOutput("output")]

        # 推理
        if client.is_model_ready("yolov5_persondog", "1"):
            results = client.infer(
                model_name="yolov5_persondog",
                inputs=inputs,
                outputs=outputs,
                model_version="1",
                client_timeout=30000
            )
            output_data = results.as_numpy("output")

            # 后处理（完全模拟 yolov5）
            label_names = ["personnew", "dognew", "catnew"]
            detected_objects = postprocess(
                output_data,
                img.shape[1], img.shape[0],
                input_shape,
                0.0,  # conf_th
                0.2,  # iou_threshold
                label_names
            )

        elapsed = time.time() - start
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

def inference_without_postprocess(img, triton_url):
    """只推理，不后处理"""
    start = time.time()

    client = grpcclient.InferenceServerClient(
        url=triton_url,
        channel_args=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    )

    try:
        # 预处理
        input_shape = [640, 640]
        input_image_buffer = preprocess(img, input_shape)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        input_image = input_image_buffer.astype(np.float32)

        inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [grpcclient.InferRequestedOutput("output")]

        # 推理
        if client.is_model_ready("yolov5_persondog", "1"):
            results = client.infer(
                model_name="yolov5_persondog",
                inputs=inputs,
                outputs=outputs,
                model_version="1",
                client_timeout=30000
            )
            output_data = results.as_numpy("output")
            # 不调用 postprocess

        elapsed = time.time() - start
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

def run_test(test_name, test_func, img, triton_url, concurrent):
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(test_func, img, triton_url)
            for _ in range(concurrent)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start_time
    success = sum(1 for r in results if r[0])
    avg_time = sum(r[1] for r in results) / len(results)

    print(f"总耗时: {elapsed:.3f}s  QPS: {concurrent / elapsed:.2f}")
    print(f"成功: {success}/{concurrent}")
    print(f"平均单次: {avg_time:.3f}s")
    print(f"并发倍数: {avg_time / elapsed:.2f}x",
          "✅ 并发" if avg_time / elapsed > 2 else "❌ 串行")

    if not results[0][0]:
        print(f"\n❌ 错误: {results[0][2]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--concurrent', type=int, default=16)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取图片: {args.image}")
        return

    print(f"图片: {img.shape}, 并发: {args.concurrent}")

    triton_url = os.getenv("TRITON_SERVER", "10.120.0.66:40008")

    # 测试 1: 不带后处理
    run_test(
        "测试 1: 推理，不调用 postprocess",
        inference_without_postprocess,
        img, triton_url, args.concurrent
    )

    # 测试 2: 带后处理
    run_test(
        "测试 2: 推理 + postprocess (完全模拟 yolov5)",
        inference_with_postprocess,
        img, triton_url, args.concurrent
    )

if __name__ == '__main__':
    main()
