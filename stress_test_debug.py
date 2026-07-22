#!/usr/bin/env python3
"""
调试版压力测试：详细记录每个阶段的耗时
"""
import asyncio
import json
import time
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kafka.message import AnalyseInputMsg, LabelEntry


async def test_single_inference_debug(image_path: str):
    """单次推理，打印详细耗时"""
    print(f"\n{'='*80}")
    print(f"调试单次推理")
    print(f"{'='*80}\n")

    # 1. 读取图片
    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    t1 = time.perf_counter()
    print(f"✓ [1] 图片读取: {(t1-t0)*1000:.1f} ms  shape={img.shape}")

    # 2. 创建消息
    t0 = time.perf_counter()
    msg = AnalyseInputMsg(
        id=int(time.time() * 1000),
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
    msg_bytes = json.dumps(msg.dict()).encode('utf-8')
    t1 = time.perf_counter()
    print(f"✓ [2] 消息构建: {(t1-t0)*1000:.1f} ms")

    # 3. 调用 process_message
    t0 = time.perf_counter()
    from kafka.handlers import process_message
    result = await process_message(msg_bytes)
    t1 = time.perf_counter()
    print(f"✓ [3] process_message: {(t1-t0)*1000:.1f} ms")

    if result:
        print(f"✓ [4] 结果: labels={len(result.get('labels', []))}")
        print(f"\n总耗时: {(t1-t0)*1000:.1f} ms")
    else:
        print(f"✗ 推理失败")


async def test_direct_triton(image_path: str):
    """直接调用 Triton，绕过 process_message"""
    print(f"\n{'='*80}")
    print(f"直接调用 Triton 测试")
    print(f"{'='*80}\n")

    # 1. 读取图片
    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    t1 = time.perf_counter()
    print(f"✓ [1] 图片读取: {(t1-t0)*1000:.1f} ms")

    # 2. 直接调用推理
    t0 = time.perf_counter()
    from api.infer.Utils.class_info import CameraInfo
    from api.infer.running import _run_analyse_sync

    camera_info = CameraInfo()
    camera_info.imgsList = [img]
    camera_info.deviceId = ""
    camera_info.presetId = "0"

    setsLabel = {"Smoking-handsmoking"}
    label_rules = {"Smoking-handsmoking": {"conf": 0.8, "iou": 0.2}}

    # 在线程池中执行（模拟真实调用）
    loop = asyncio.get_event_loop()
    from api.infer.running import logic_executor
    result = await loop.run_in_executor(
        logic_executor,
        _run_analyse_sync,
        camera_info,
        setsLabel,
        label_rules
    )
    t1 = time.perf_counter()

    print(f"✓ [2] _run_analyse_sync: {(t1-t0)*1000:.1f} ms")
    print(f"✓ [3] 结果数量: {len(result)}")
    print(f"\n总耗时: {(t1-t0)*1000:.1f} ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--mode", choices=["process", "direct"], default="process")
    args = parser.parse_args()

    if args.mode == "process":
        asyncio.run(test_single_inference_debug(args.image))
    else:
        asyncio.run(test_direct_triton(args.image))
