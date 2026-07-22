#!/usr/bin/env python3
"""
测试在推理循环内部调用 is_model_ready() 是否导致串行
"""
import time
import cv2
import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as grpcclient

def prepare_input(img):
    """简单预处理"""
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

def inference_with_ready_check(img, triton_url):
    """每次推理前都调用 is_model_ready()（模拟 yolov5() 的行为）"""
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
        input_data = prepare_input(img)

        # 关键：在推理前调用 is_model_ready()（和 yolov5() 一样）
        if client.is_model_ready("yolov5_persondog", "1"):
            inputs = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")]
            inputs[0].set_data_from_numpy(input_data)

            outputs = [grpcclient.InferRequestedOutput("output0")]

            result = client.infer(
                model_name="yolov5_persondog",
                inputs=inputs,
                outputs=outputs,
                model_version="1"
            )

            output = result.as_numpy("output0")

        elapsed = time.time() - start
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

def inference_without_ready_check(img, triton_url):
    """不调用 is_model_ready()，直接推理"""
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
        input_data = prepare_input(img)

        # 直接推理，不检查 ready
        inputs = [grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput("output0")]

        result = client.infer(
            model_name="yolov5_persondog",
            inputs=inputs,
            outputs=outputs,
            model_version="1"
        )

        output = result.as_numpy("output0")

        elapsed = time.time() - start
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

def run_test(test_name, test_func, img, triton_url, concurrent):
    """运行测试"""
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

    # 打印第一个错误
    if not results[0][0]:
        print(f"\n❌ 错误示例: {results[0][2]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--concurrent', type=int, default=16)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取图片: {args.image}")
        return

    print(f"图片: {img.shape}, Triton: {os.getenv('TRITON_SERVER')}, 并发: {args.concurrent}")

    triton_url = os.getenv("TRITON_SERVER", "10.120.0.66:40008")

    # 测试 1: 每次调用 is_model_ready()（模拟 yolov5()）
    run_test(
        "测试 1: 推理前调用 is_model_ready() (模拟 yolov5 函数行为)",
        inference_with_ready_check,
        img, triton_url, args.concurrent
    )

    # 测试 2: 不调用 is_model_ready()
    run_test(
        "测试 2: 直接推理，不检查 ready",
        inference_without_ready_check,
        img, triton_url, args.concurrent
    )

if __name__ == '__main__':
    main()
