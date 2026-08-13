#!/usr/bin/env python3
"""
性能分析脚本：追踪 Smoking-handsmoking 每个阶段的耗时
"""
import asyncio
import json
import time
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.message import AnalyseInputMsg, LabelEntry


async def profile_single_inference(image_path: str):
    """单次推理性能分析"""
    print(f"\n{'='*80}")
    print(f"Smoking-handsmoking 性能分析")
    print(f"{'='*80}\n")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    print(f"✓ 图片读取成功: {img.shape}\n")

    # 创建消息
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

    # 调用推理
    print("开始推理...\n")
    t_start = time.perf_counter()

    from inference.handler import process_message
    msg_bytes = json.dumps(msg.dict()).encode('utf-8')
    result = await process_message(msg_bytes)

    t_end = time.perf_counter()

    print(f"\n{'='*80}")
    print(f"推理完成")
    print(f"{'='*80}")
    print(f"总耗时: {(t_end - t_start):.3f} 秒")

    if result:
        labels = result.get('labels', [])
        print(f"检测结果: {len(labels)} 个标签")
        print(f"modelStartTime: {result.get('modelStartTime', 'N/A')}")
        print(f"modelTime: {result.get('modelTime', 'N/A')}")
    else:
        print("推理失败")

    print(f"{'='*80}\n")


async def profile_direct_triton(image_path: str):
    """直接测试 personnew + zuidiaoyan，绕过 process_message"""
    print(f"\n{'='*80}")
    print(f"直接 Triton 推理测试 (绕过 process_message)")
    print(f"{'='*80}\n")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    from api.infer.Utils.class_info import CameraInfo
    from api.infer.running import _run_analyse_sync, logic_executor

    # 阶段 1：测试 personnew 推理
    print("测试阶段 1: personnew 推理")
    camera_info = CameraInfo()
    camera_info.imgsList = [img]
    camera_info.deviceId = ""
    camera_info.presetId = "0"

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        logic_executor,
        _run_analyse_sync,
        camera_info,
        {"personnew"},
        {"personnew": {"conf": 0.1, "iou": 0.2}}
    )
    t1 = time.perf_counter()

    personnew_count = sum(len(v.get('bbox', [])) for v in result.values())
    print(f"✓ personnew 推理耗时: {(t1-t0):.3f} 秒")
    print(f"  检测到 {personnew_count} 个人\n")

    # 阶段 2：测试 zuidiaoyan 推理
    print("测试阶段 2: zuidiaoyan 推理 (在 personnew 的基础上)")

    if personnew_count == 0:
        print("⚠️  没有检测到人，跳过 zuidiaoyan 测试\n")
        return

    t0 = time.perf_counter()
    result2 = await loop.run_in_executor(
        logic_executor,
        _run_analyse_sync,
        camera_info,
        {"zuidiaoyan"},
        {"zuidiaoyan": {"conf": 0.8, "iou": 0.2}}
    )
    t1 = time.perf_counter()

    zuidiaoyan_count = sum(len(v.get('bbox', [])) for v in result2.values())
    print(f"✓ zuidiaoyan 推理耗时: {(t1-t0):.3f} 秒")
    print(f"  检测到 {zuidiaoyan_count} 个目标\n")

    print(f"{'='*80}")
    print(f"总结")
    print(f"{'='*80}")
    print(f"如果这两个阶段都很快 (<1秒)，说明 Triton 本身没问题")
    print(f"如果仍然很慢，说明是 Triton 服务端配置问题")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--mode", choices=["full", "direct"], default="full",
                       help="full=完整流程, direct=直接 Triton")
    args = parser.parse_args()

    if args.mode == "full":
        asyncio.run(profile_single_inference(args.image))
    else:
        asyncio.run(profile_direct_triton(args.image))
