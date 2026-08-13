#!/usr/bin/env python3
"""
并发度实时监控
"""
import asyncio
import json
import time
import sys
from pathlib import Path
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.message import AnalyseInputMsg, LabelEntry

# 全局监控
active_tasks = 0
max_concurrent = 0
lock = threading.Lock()


class Monitor:
    def __init__(self):
        self.events = []

    def log(self, event_type, task_id):
        with lock:
            global active_tasks, max_concurrent
            if event_type == "start":
                active_tasks += 1
                max_concurrent = max(max_concurrent, active_tasks)
            elif event_type == "end":
                active_tasks -= 1

            self.events.append({
                "time": time.perf_counter(),
                "type": event_type,
                "task_id": task_id,
                "active": active_tasks
            })


monitor = Monitor()


async def traced_inference(image_path: str, task_id: int):
    """带追踪的推理"""
    monitor.log("start", task_id)

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

    try:
        from inference.handler import process_message
        msg_bytes = json.dumps(msg.dict()).encode('utf-8')
        result = await process_message(msg_bytes)
        t1 = time.perf_counter()

        monitor.log("end", task_id)
        return {"task_id": task_id, "duration": t1 - t0, "success": True}
    except Exception as e:
        monitor.log("end", task_id)
        return {"task_id": task_id, "duration": 0, "success": False, "error": str(e)}


async def run_concurrent_test(image_path: str, concurrent: int):
    """运行并发测试"""
    print(f"\n{'='*80}")
    print(f"并发度追踪测试")
    print(f"{'='*80}")
    print(f"并发数: {concurrent}")
    print(f"{'='*80}\n")

    tasks = [traced_inference(image_path, i) for i in range(concurrent)]

    print(f"提交 {concurrent} 个并发任务...")
    start_time = time.perf_counter()

    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()

    print(f"\n{'='*80}")
    print(f"测试完成")
    print(f"{'='*80}")
    print(f"总耗时: {end_time - start_time:.3f} 秒")
    print(f"成功: {sum(1 for r in results if r['success'])}/{concurrent}")
    print(f"最大并发度: {max_concurrent}")

    durations = [r["duration"] for r in results if r["success"]]
    if durations:
        print(f"\n延迟统计:")
        print(f"  最小: {min(durations):.3f}s")
        print(f"  最大: {max(durations):.3f}s")
        print(f"  平均: {sum(durations)/len(durations):.3f}s")

    # 分析并发模式
    print(f"\n{'='*80}")
    print(f"并发度变化")
    print(f"{'='*80}")

    start_t = monitor.events[0]["time"]
    for i, event in enumerate(monitor.events[:20]):  # 只显示前 20 个事件
        elapsed = event["time"] - start_t
        print(f"{elapsed:6.3f}s: 任务 {event['task_id']:2d} {event['type']:5s} → 活跃数: {event['active']}")

    if len(monitor.events) > 20:
        print(f"... (省略 {len(monitor.events) - 20} 个事件)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    asyncio.run(run_concurrent_test(args.image, args.concurrent))
