#!/usr/bin/env python3
"""
Triton 连接池监控脚本
"""
import asyncio
import json
import time
import sys
from pathlib import Path
import cv2
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.message import AnalyseInputMsg, LabelEntry
from tools.init import tritonServer

# 全局计数器
active_requests = 0
lock = threading.Lock()


async def monitored_inference(image_path: str, task_id: int):
    """带监控的推理"""
    global active_requests

    with lock:
        active_requests += 1
        current = active_requests

    try:
        msg = AnalyseInputMsg(
            id=int(time.time() * 1000) + task_id,
            path=image_path,
            eos=False,
            labels=[
                LabelEntry(
                    alarmTypeId=23,
                    inferLabels="Smoking-handsmoking",
                    labelDetails='[{"conf":0.8,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
                )
            ],
            questions=[],
            vector=False
        )

        t0 = time.perf_counter()

        from inference.handler import process_message
        msg_bytes = json.dumps(msg.dict()).encode('utf-8')
        result = await process_message(msg_bytes)

        t1 = time.perf_counter()

        return {
            "task_id": task_id,
            "duration": t1 - t0,
            "peak_concurrent": current,
            "success": result is not None
        }
    finally:
        with lock:
            active_requests -= 1


async def monitor_triton_pool():
    """监控 Triton 连接池状态"""
    try:
        pool = tritonServer._pool
        while True:
            await asyncio.sleep(0.5)
            qsize = pool.qsize()
            maxsize = pool.maxsize
            in_use = maxsize - qsize
            print(f"[监控] Triton 连接池: {in_use}/{maxsize} 使用中, {qsize} 空闲")
    except Exception as e:
        print(f"监控失败: {e}")


async def stress_test_with_monitor(image_path: str, concurrent: int, duration: int):
    """带监控的压力测试"""
    print(f"\n{'='*80}")
    print(f"带监控的压力测试")
    print(f"{'='*80}")
    print(f"并发数: {concurrent}")
    print(f"测试时长: {duration} 秒")
    print(f"Triton 连接池大小: {tritonServer._pool.maxsize}")
    print(f"{'='*80}\n")

    # 启动监控任务
    monitor_task = asyncio.create_task(monitor_triton_pool())

    start_time = time.perf_counter()
    end_time = start_time + duration

    results = []
    task_id = 0
    pending_tasks = set()

    try:
        while time.perf_counter() < end_time or pending_tasks:
            # 保持并发数
            while len(pending_tasks) < concurrent and time.perf_counter() < end_time:
                task = asyncio.create_task(monitored_inference(image_path, task_id))
                pending_tasks.add(task)
                task_id += 1

            # 等待任何一个任务完成
            if pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1
                )

                for task in done:
                    result = await task
                    results.append(result)

                    if len(results) % 5 == 0:
                        avg_duration = sum(r["duration"] for r in results) / len(results)
                        print(f"进度: {len(results)} 完成, 平均耗时: {avg_duration:.3f}s")

    except KeyboardInterrupt:
        print("\n中断测试...")
    finally:
        monitor_task.cancel()

    # 统计
    total_elapsed = time.perf_counter() - start_time

    print(f"\n{'='*80}")
    print(f"测试结果")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed:.3f} 秒")
    print(f"总请求: {len(results)}")
    print(f"QPS: {len(results) / total_elapsed:.2f}")

    if results:
        durations = [r["duration"] for r in results]
        durations.sort()
        print(f"\n延迟统计:")
        print(f"  最小值: {min(durations):.3f}s")
        print(f"  最大值: {max(durations):.3f}s")
        print(f"  平均值: {sum(durations)/len(durations):.3f}s")
        print(f"  中位数: {durations[len(durations)//2]:.3f}s")

        peak = max(r["peak_concurrent"] for r in results)
        print(f"\n最大并发数: {peak}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrent", type=int, default=16)
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(stress_test_with_monitor(args.image, args.concurrent, args.duration))
