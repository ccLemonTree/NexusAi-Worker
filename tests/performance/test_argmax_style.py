#!/usr/bin/env python3
"""
对比 yolov5 的循环 argmax vs yolov11 的向量化 argmax
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def yolov5_style_argmax():
    """yolov5 风格：循环调用 argmax"""
    start = time.time()
    boxes1 = np.random.rand(25200, 85).astype(np.float32)

    zzz = []
    for i in range(len(boxes1)):
        boxe = list(boxes1[i, 0:5])
        boxe.append(int(np.argmax(boxes1[i, 5:])))  # 循环调用
        zzz.append(np.array(boxe))
    boxes = np.array(zzz)

    elapsed = time.time() - start
    return elapsed

def yolov11_style_argmax():
    """yolov11 风格：向量化 argmax"""
    start = time.time()
    boxes1 = np.random.rand(25200, 85).astype(np.float32)

    class_ids = np.argmax(boxes1[:, 5:], axis=1)  # 向量化
    confidences = np.max(boxes1[:, 5:], axis=1)

    elapsed = time.time() - start
    return elapsed

def run_test(name, func, concurrent):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: func(), range(concurrent)))
    elapsed = time.time() - start_time

    avg = sum(results) / len(results)

    print(f"总耗时: {elapsed:.3f}s")
    print(f"平均单次: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x",
          "✅ 并发" if avg / elapsed > 2 else "❌ 串行")

if __name__ == '__main__':
    print("对比 yolov5 循环风格 vs yolov11 向量化风格")
    run_test("yolov5 风格 (循环 argmax)", yolov5_style_argmax, 16)
    run_test("yolov11 风格 (向量化 argmax)", yolov11_style_argmax, 16)
