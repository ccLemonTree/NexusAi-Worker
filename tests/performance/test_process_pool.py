#!/usr/bin/env python3
"""
测试使用进程池是否能解决 GIL 问题
"""
import time
import cv2
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

def cpu_intensive():
    """CPU 密集型操作"""
    start = time.time()
    total = 0
    for i in range(10000000):
        total += i * i
    elapsed = time.time() - start
    return elapsed

def worker(_):
    """可序列化的 worker 函数"""
    return cpu_intensive()

def run_test(name, executor_class, concurrent):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    start_time = time.time()
    with executor_class(max_workers=concurrent) as executor:
        results = list(executor.map(worker, range(concurrent)))
    elapsed = time.time() - start_time

    avg = sum(results) / len(results)

    print(f"总耗时: {elapsed:.3f}s")
    print(f"平均单次: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x",
          "✅ 并发" if avg / elapsed > 2 else "❌ 串行")

if __name__ == '__main__':
    print("对比线程池 vs 进程池")
    run_test("线程池 (受 GIL 限制)", ThreadPoolExecutor, 16)
    run_test("进程池 (不受 GIL 限制)", ProcessPoolExecutor, 16)
