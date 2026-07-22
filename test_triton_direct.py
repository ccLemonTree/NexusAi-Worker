#!/usr/bin/env python3
"""
直接测试 Triton 推理的并发性
"""
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from tools.init import tritonServer

def single_triton_inference(image_path: str, task_id: int):
    """单次 Triton 推理"""
    img = cv2.imread(image_path)

    t0 = time.perf_counter()
    result = tritonServer.run("yolov5_persondog", img, label_to_detect={"personnew": {"conf": 0.1, "iou": 0.2}})
    t1 = time.perf_counter()

    return {"task_id": task_id, "duration": t1 - t0, "count": len(result)}


def test_concurrent_triton(image_path: str, concurrent: int):
    """并发测试 Triton"""
    print(f"\n{'='*80}")
    print(f"直接测试 Triton 并发性")
    print(f"{'='*80}")
    print(f"并发数: {concurrent}")
    print(f"模型: yolov5_persondog (personnew)")
    print(f"{'='*80}\n")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(single_triton_inference, image_path, i) for i in range(concurrent)]

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"任务 {result['task_id']:2d} 完成: {result['duration']:.3f}s, 检测到 {result['count']} 个")

    end_time = time.perf_counter()

    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}")
    print(f"总耗时: {end_time - start_time:.3f} 秒")
    print(f"QPS: {concurrent / (end_time - start_time):.2f}")

    durations = [r["duration"] for r in results]
    print(f"\n推理耗时:")
    print(f"  最小: {min(durations):.3f}s")
    print(f"  最大: {max(durations):.3f}s")
    print(f"  平均: {sum(durations)/len(durations):.3f}s")

    # 判断是否并发
    total_inference_time = sum(durations)
    wall_time = end_time - start_time

    print(f"\n并发分析:")
    print(f"  所有推理时间总和: {total_inference_time:.3f}s")
    print(f"  实际墙上时间: {wall_time:.3f}s")
    print(f"  并发倍数: {total_inference_time / wall_time:.2f}x")

    if total_inference_time / wall_time > concurrent * 0.8:
        print(f"  ❌ 串行执行！没有真正并发")
    elif total_inference_time / wall_time > 1.5:
        print(f"  ✅ 有并发，但不是完全并发")
    else:
        print(f"  ⚠️  完全串行或只有少量并发")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    test_concurrent_triton(args.image, args.concurrent)
