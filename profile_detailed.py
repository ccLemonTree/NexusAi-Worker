#!/usr/bin/env python3
"""
详细的性能分析：追踪每一步的耗时
"""
import time
import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.infer.Utils.class_info import CameraInfo
from api.infer.running import analyseRun
from tools.init import tritonServer, cfg


def profile_personnew(image_path: str):
    """详细分析 personnew 推理"""
    print(f"\n{'='*80}")
    print("personnew 推理详细分析")
    print(f"{'='*80}\n")

    # 1. 读取图片
    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    t1 = time.perf_counter()
    print(f"1. 图片读取: {(t1-t0)*1000:.1f} ms, shape={img.shape}")

    # 2. 构造 CameraInfo
    t0 = time.perf_counter()
    camera_info = CameraInfo()
    camera_info.imgsList = [img, img]
    camera_info.deviceId = ""
    camera_info.presetId = "0"
    t1 = time.perf_counter()
    print(f"2. CameraInfo 构造: {(t1-t0)*1000:.1f} ms")

    # 3. 调用 analyseRun (personnew)
    t0 = time.perf_counter()
    setsLabel = ["personnew"]
    label_rules = {"personnew": {"conf": 0.1, "iou": 0.2}}
    result = analyseRun(setsLabel, [img, img], camera_info, label_rules)
    t1 = time.perf_counter()
    print(f"3. analyseRun (personnew): {(t1-t0)*1000:.1f} ms")
    print(f"   检测到 {len(result)} 个人")

    # 4. 切图（模拟 Smoking.py 的操作）
    if len(result) > 0:
        t0 = time.perf_counter()
        cut_images = []
        for box in result:
            cut_img = img[box.y1:box.y2, box.x1:box.x2]
            cut_images.append(cut_img)
        t1 = time.perf_counter()
        print(f"4. 切图 {len(cut_images)} 个: {(t1-t0)*1000:.1f} ms")

        # 5. 对每个切图推理 zuidiaoyan
        t0 = time.perf_counter()
        for i, cut_img in enumerate(cut_images):
            camera_info2 = CameraInfo()
            camera_info2.imgsList = [cut_img, cut_img]
            result2 = analyseRun(["zuidiaoyan"], [cut_img, cut_img], camera_info2,
                                {"zuidiaoyan": {"conf": 0.8, "iou": 0.2}})
            print(f"   切图 {i+1} zuidiaoyan 推理: {len(result2)} 个结果")
        t1 = time.perf_counter()
        print(f"5. {len(cut_images)} 个 zuidiaoyan 推理总耗时: {(t1-t0)*1000:.1f} ms")
        print(f"   平均每个: {(t1-t0)*1000/len(cut_images):.1f} ms")

    print(f"{'='*80}\n")


def profile_zuidiaoyan_direct(image_path: str):
    """直接在大图上推理 zuidiaoyan"""
    print(f"\n{'='*80}")
    print("zuidiaoyan 直接在大图推理")
    print(f"{'='*80}\n")

    img = cv2.imread(image_path)
    camera_info = CameraInfo()
    camera_info.imgsList = [img, img]

    t0 = time.perf_counter()
    result = analyseRun(["zuidiaoyan"], [img, img], camera_info,
                       {"zuidiaoyan": {"conf": 0.8, "iou": 0.2}})
    t1 = time.perf_counter()

    print(f"大图 (shape={img.shape}) zuidiaoyan 推理: {(t1-t0)*1000:.1f} ms")
    print(f"检测到 {len(result)} 个结果")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", choices=["personnew", "zuidiaoyan"], default="personnew")
    args = parser.parse_args()

    if args.mode == "personnew":
        profile_personnew(args.image)
    else:
        profile_zuidiaoyan_direct(args.image)
