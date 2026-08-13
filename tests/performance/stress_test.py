#!/usr/bin/env python3
"""
压力测试脚本：测试 Triton 推理吞吐量

使用方法：
    python tests/performance/stress_test.py --image test.jpg --duration 10 --concurrent 16
    python tests/performance/stress_test.py --image test.jpg --count 100 --concurrent 8

测试场景：
    1. 测试所有 5 个标签（Smoking-handsmoking, 26_smoke, Hot-work, 26_fire + VLM）
    2. 仅测试 4 个小模型标签（排除 VLM）
"""
import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.message import AnalyseInputMsg, LabelEntry, QuestionEntry
from inference.handler import process_message
import cv2
import numpy as np
import json


def create_test_message(image_path: str, test_labels: str) -> AnalyseInputMsg:
    """创建测试消息"""
    if test_labels == "all":
        # 测试所有 5 个标签
        labels = [
            LabelEntry(
                alarmTypeId=23,
                inferLabels="Smoking-handsmoking",
                labelDetails='[{"conf":0.8,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
            ),
            LabelEntry(
                alarmTypeId=26,
                inferLabels="26_smoke",
                labelDetails='[{"conf":0.62,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
            ),
            LabelEntry(
                alarmTypeId=40000001,
                inferLabels="Hot-work",
                labelDetails='[{"conf":0.7,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
            ),
            LabelEntry(
                alarmTypeId=24,
                inferLabels="26_fire",
                labelDetails='[{"conf":0.62,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
            ),
        ]
        questions = [
            QuestionEntry(
                system="",
                question="烟雾检测-大模型（你是一名消防员，能仔细分辨图像中存在的火灾引发的烟雾，排除水蒸气、烟囱冒烟的行为。）；火情检测-大模型（你是一名消防员，能仔细分辨图像中存在的火灾明火,排除灯光、反光影响。）；"
            )
        ]
    else:
        # 仅测试 1 个小模型标签（单独测试 Smoking-handsmoking）
        labels = [
            LabelEntry(
                alarmTypeId=23,
                inferLabels="Smoking-handsmoking",
                labelDetails='[{"conf":0.8,"desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]'
            )
        ]
        questions = []  # 不调用 VLM

    return AnalyseInputMsg(
        id=int(time.time() * 1000),
        path=image_path,
        eos=False,  # 使用本地文件
        labels=labels,
        questions=questions,
        vector=False
    )


async def run_single_inference(msg: AnalyseInputMsg, task_id: int) -> Dict[str, Any]:
    """执行单次推理并返回耗时统计"""
    start_time = time.perf_counter()

    try:
        # 将消息转换为 JSON bytes 格式（process_message 接收 bytes）
        msg_bytes = json.dumps(msg.dict()).encode('utf-8')
        result = await process_message(msg_bytes)
        end_time = time.perf_counter()

        return {
            "task_id": task_id,
            "success": result is not None,
            "duration": end_time - start_time,
            "label_count": len(result.get("labels", [])) if result else 0,
            "error": None
        }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "task_id": task_id,
            "success": False,
            "duration": end_time - start_time,
            "label_count": 0,
            "error": str(e)
        }


async def stress_test_duration(image_path: str, duration: int, concurrent: int, test_labels: str):
    """持续时间测试：在指定时间内持续推理"""
    print(f"\n{'='*80}")
    print(f"持续时间压力测试")
    print(f"{'='*80}")
    print(f"测试图片: {image_path}")
    print(f"测试时长: {duration} 秒")
    print(f"并发数: {concurrent}")
    print(f"测试标签: {'所有5个标签(含VLM)' if test_labels == 'all' else '仅4个小模型标签'}")
    print(f"{'='*80}\n")

    # 验证图片存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件不存在: {image_path}")
        return

    # 读取图片验证
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 错误：无法读取图片: {image_path}")
        return
    print(f"✓ 图片读取成功: {img.shape}\n")

    start_time = time.perf_counter()
    end_time = start_time + duration

    results = []
    task_id = 0
    pending_tasks = set()

    try:
        while time.perf_counter() < end_time or pending_tasks:
            # 保持并发数
            while len(pending_tasks) < concurrent and time.perf_counter() < end_time:
                msg = create_test_message(image_path, test_labels)
                task = asyncio.create_task(run_single_inference(msg, task_id))
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

                    # 实时显示进度
                    if len(results) % 10 == 0:
                        elapsed = time.perf_counter() - start_time
                        qps = len(results) / elapsed if elapsed > 0 else 0
                        success_count = sum(1 for r in results if r["success"])
                        print(f"进度: {len(results)} 次推理完成, 成功 {success_count}, 当前QPS: {qps:.2f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，正在停止测试...\n")
        # 取消所有待处理的任务
        for task in pending_tasks:
            task.cancel()

    # 统计结果
    total_elapsed = time.perf_counter() - start_time
    print_statistics(results, total_elapsed)


async def stress_test_count(image_path: str, count: int, concurrent: int, test_labels: str):
    """固定次数测试：推理指定次数"""
    print(f"\n{'='*80}")
    print(f"固定次数压力测试")
    print(f"{'='*80}")
    print(f"测试图片: {image_path}")
    print(f"推理次数: {count}")
    print(f"并发数: {concurrent}")
    print(f"测试标签: {'所有5个标签(含VLM)' if test_labels == 'all' else '仅4个小模型标签'}")
    print(f"{'='*80}\n")

    # 验证图片存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件不存在: {image_path}")
        return

    # 读取图片验证
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 错误：无法读取图片: {image_path}")
        return
    print(f"✓ 图片读取成功: {img.shape}\n")

    start_time = time.perf_counter()

    # 创建所有任务
    tasks = []
    for i in range(count):
        msg = create_test_message(image_path, test_labels)
        task = asyncio.create_task(run_single_inference(msg, i))
        tasks.append(task)

        # 控制并发数
        if len(tasks) >= concurrent:
            done = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = list(done[1])  # 保留未完成的任务

    # 等待所有任务完成
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        result = await task
        results.append(result)

        # 实时显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            elapsed = time.perf_counter() - start_time
            qps = len(results) / elapsed if elapsed > 0 else 0
            success_count = sum(1 for r in results if r["success"])
            print(f"进度: {len(results)}/{count}, 成功 {success_count}, 当前QPS: {qps:.2f}")

    # 统计结果
    total_elapsed = time.perf_counter() - start_time
    print_statistics(results, total_elapsed)


def print_statistics(results: List[Dict[str, Any]], total_elapsed: float):
    """打印统计结果"""
    total_count = len(results)
    success_count = sum(1 for r in results if r["success"])
    failed_count = total_count - success_count

    durations = [r["duration"] for r in results if r["success"]]

    print(f"\n{'='*80}")
    print(f"测试结果统计")
    print(f"{'='*80}")
    print(f"总耗时: {total_elapsed:.3f} 秒")
    print(f"总请求数: {total_count}")
    print(f"成功: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"失败: {failed_count} ({failed_count/total_count*100:.1f}%)")
    print(f"\n吞吐量:")
    print(f"  QPS (每秒推理图片数): {total_count / total_elapsed:.2f}")

    if durations:
        durations.sort()
        print(f"\n延迟统计 (秒):")
        print(f"  最小值: {min(durations):.3f}")
        print(f"  最大值: {max(durations):.3f}")
        print(f"  平均值: {sum(durations)/len(durations):.3f}")
        print(f"  中位数: {durations[len(durations)//2]:.3f}")
        print(f"  P95: {durations[int(len(durations)*0.95)]:.3f}")
        print(f"  P99: {durations[int(len(durations)*0.99)]:.3f}")

    # 显示错误信息
    if failed_count > 0:
        print(f"\n错误列表:")
        error_summary = {}
        for r in results:
            if not r["success"]:
                error = r["error"]
                error_summary[error] = error_summary.get(error, 0) + 1

        for error, count in sorted(error_summary.items(), key=lambda x: -x[1]):
            print(f"  [{count}次] {error}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Triton 推理压力测试")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--duration", type=int, help="持续测试时长（秒）")
    parser.add_argument("--count", type=int, help="推理次数（与 duration 二选一）")
    parser.add_argument("--concurrent", type=int, default=16, help="并发数（默认16）")
    parser.add_argument(
        "--labels",
        choices=["all", "small"],
        default="all",
        help="测试标签：all=所有5个标签(含VLM), small=仅4个小模型标签（默认all）"
    )

    args = parser.parse_args()

    if args.duration is None and args.count is None:
        parser.error("必须指定 --duration 或 --count 之一")

    if args.duration is not None and args.count is not None:
        parser.error("--duration 和 --count 不能同时指定")

    # 运行测试
    if args.duration:
        asyncio.run(stress_test_duration(args.image, args.duration, args.concurrent, args.labels))
    else:
        asyncio.run(stress_test_count(args.image, args.count, args.concurrent, args.labels))


if __name__ == "__main__":
    main()
