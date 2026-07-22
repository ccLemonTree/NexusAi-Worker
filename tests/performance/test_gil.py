#!/usr/bin/env python3
"""
测试纯 Python CPU 密集型操作在多线程下的表现
"""
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def cpu_intensive_python():
    """纯 Python CPU 密集型操作"""
    start = time.time()
    total = 0
    for i in range(10000000):
        total += i * i
    elapsed = time.time() - start
    return True, elapsed

def cpu_intensive_list_ops():
    """Python 列表操作"""
    start = time.time()
    data = [i for i in range(1000000)]
    sorted_data = sorted(data, reverse=True)
    elapsed = time.time() - start
    return True, elapsed

def run_test(name, func, concurrent):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"并发: {concurrent}")
    print(f"{'='*80}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: func(), range(concurrent)))
    elapsed = time.time() - start_time

    success = sum(1 for r in results if r[0])
    avg = sum(r[1] for r in results) / len(results)

    print(f"总耗时: {elapsed:.3f}s")
    print(f"成功: {success}/{concurrent}")
    print(f"平均单次: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x",
          "✅ 并发" if avg / elapsed > 2 else "❌ 串行")

if __name__ == '__main__':
    print("测试 Python GIL 对 CPU 密集型操作的影响")
    run_test("测试 1: 纯 Python 计算", cpu_intensive_python, 16)
    run_test("测试 2: Python 列表排序", cpu_intensive_list_ops, 16)
