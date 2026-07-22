#!/usr/bin/env python3
"""
测试 cv2.resize 和 cv2.cvtColor 在多线程下是否有性能问题
"""
import time
import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_with_cv2(img):
    """使用 cv2 进行预处理（模拟 yolov5 的 preprocess）"""
    start = time.time()

    # 模拟 yolov5 的 letterbox resize
    img_h, img_w, _ = img.shape
    new_h, new_w = 640, 640
    offset_h, offset_w = 0, 0

    if (new_w / img_w) <= (new_h / img_h):
        new_h = int(img_h * new_w / img_w)
        offset_h = (640 - new_h) // 2
    else:
        new_w = int(img_w * new_h / img_h)
        offset_w = (640 - new_w) // 2

    resized = cv2.resize(img, (new_w, new_h))
    result = np.full((640, 640, 3), 127, dtype=np.uint8)
    result[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized

    # cvtColor
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # transpose 和 normalize
    result = result.transpose((2, 0, 1)).astype(np.float32)
    result /= 255.0

    elapsed = time.time() - start
    return True, elapsed

def preprocess_without_cv2(img):
    """不使用 cv2，只用 numpy"""
    start = time.time()

    # 简单 resize（用切片模拟）
    result = img[:640, :640, :].copy()

    # 手动 BGR to RGB
    result = result[:, :, ::-1]

    # transpose 和 normalize
    result = result.transpose((2, 0, 1)).astype(np.float32)
    result /= 255.0

    elapsed = time.time() - start
    return True, elapsed

def run_test(test_name, test_func, img, concurrent):
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(test_func, img)
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

    # 测试 1: 使用 cv2（模拟真实 yolov5 预处理）
    run_test(
        "测试 1: 使用 cv2.resize + cv2.cvtColor (真实预处理)",
        preprocess_with_cv2,
        img, args.concurrent
    )

    # 测试 2: 纯 numpy
    run_test(
        "测试 2: 纯 numpy 操作",
        preprocess_without_cv2,
        img, args.concurrent
    )

if __name__ == '__main__':
    main()
