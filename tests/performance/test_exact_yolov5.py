#!/usr/bin/env python3
"""
完全模拟真实 yolov5() 函数的调用，使用正确的输入输出名称
"""
import time
import cv2
import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as grpcclient

def preprocess(img, input_shape):
    """完全照搬 yolov5 的预处理"""
    img_h, img_w, _ = img.shape
    new_h, new_w = input_shape[0], input_shape[1]
    offset_h, offset_w = 0, 0
    if (new_w / img_w) <= (new_h / img_h):
        new_h = int(img_h * new_w / img_w)
        offset_h = (input_shape[0] - new_h) // 2
    else:
        new_w = int(img_w * new_h / img_h)
        offset_w = (input_shape[1] - new_w) // 2
    resized = cv2.resize(img, (new_w, new_h))
    result = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
    result[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = result.transpose((2, 0, 1)).astype(np.float32)
    result /= 255.0
    return result

def single_inference(img, triton_url):
    """完全模拟 yolov5() 的推理流程"""
    start = time.time()

    # 创建独立客户端
    client = grpcclient.InferenceServerClient(
        url=triton_url,
        channel_args=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    )

    try:
        # 预处理（完全照搬 yolov5）
        input_shape = [640, 640]
        input_image_buffer = preprocess(img, input_shape)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        input_image = input_image_buffer.astype(np.float32)

        # 构建输入输出（使用正确的名称）
        inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [grpcclient.InferRequestedOutput("output")]

        # 推理（完全模拟 yolov5）
        if client.is_model_ready("yolov5_persondog", "1"):
            results = client.infer(
                model_name="yolov5_persondog",
                inputs=inputs,
                outputs=outputs,
                model_version="1",
                client_timeout=30000  # 和 yolov5 一样的超时
            )
            output_data = results.as_numpy("output")

        elapsed = time.time() - start
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

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

    print(f"\n{'='*80}")
    print(f"完全模拟 yolov5() 函数 (预处理 + is_model_ready + infer + 正确的输入输出)")
    print(f"{'='*80}")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = [
            executor.submit(single_inference, img, triton_url)
            for _ in range(args.concurrent)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start_time

    success = sum(1 for r in results if r[0])
    avg_time = sum(r[1] for r in results) / len(results)

    print(f"总耗时: {elapsed:.3f}s  QPS: {args.concurrent / elapsed:.2f}")
    print(f"成功: {success}/{args.concurrent}")
    print(f"平均单次: {avg_time:.3f}s")
    print(f"并发倍数: {avg_time / elapsed:.2f}x",
          "✅ 并发" if avg_time / elapsed > 2 else "❌ 串行")

    if not results[0][0]:
        print(f"\n❌ 错误: {results[0][2]}")

if __name__ == '__main__':
    main()
