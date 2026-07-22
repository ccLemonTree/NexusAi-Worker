#!/usr/bin/env python3
"""
逐步拆解 NMS，找出慢在哪里
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def test_step1_argmax_only():
    """步骤 1: 只做 argmax（已优化）"""
    start = time.time()
    prediction = np.random.rand(25200, 85).astype(np.float32)
    boxes1 = prediction[prediction[:, 4] >= 0.0]

    # 向量化 argmax
    class_ids = np.argmax(boxes1[:, 5:], axis=1, keepdims=True)
    boxes = np.concatenate([boxes1[:, :5], class_ids], axis=1)

    elapsed = time.time() - start
    return elapsed

def test_step2_with_coordinate_transform():
    """步骤 2: argmax + 坐标转换 + clip"""
    start = time.time()
    prediction = np.random.rand(25200, 85).astype(np.float32)
    boxes1 = prediction[prediction[:, 4] >= 0.0]

    class_ids = np.argmax(boxes1[:, 5:], axis=1, keepdims=True)
    boxes = np.concatenate([boxes1[:, :5], class_ids], axis=1)

    # 坐标转换（简化版）
    boxes[:, 0] = np.clip(boxes[:, 0], 0, 1920)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, 1080)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, 1920)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, 1080)

    # argsort
    boxes = boxes[np.argsort(-boxes[:, 4])]

    elapsed = time.time() - start
    return elapsed

def test_step3_with_nms_loop():
    """步骤 3: 完整 NMS（包括 while 循环）"""
    start = time.time()
    prediction = np.random.rand(25200, 85).astype(np.float32)
    boxes1 = prediction[prediction[:, 4] >= 0.0]

    class_ids = np.argmax(boxes1[:, 5:], axis=1, keepdims=True)
    boxes = np.concatenate([boxes1[:, :5], class_ids], axis=1)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, 1920)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, 1080)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, 1920)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, 1080)
    boxes = boxes[np.argsort(-boxes[:, 4])]

    # NMS while 循环
    keep_boxes = []
    nms_thres = 0.5
    while boxes.shape[0]:
        # IoU 计算
        box1 = np.expand_dims(boxes[0, :4], 0)
        box2 = boxes[:, :4]

        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        large_overlap = iou > nms_thres

        label_match = boxes[0, -1] == boxes[:, -1]
        invalid = large_overlap & label_match
        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]

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
    print("逐步测试 NMS 各部分")
    run_test("步骤 1: 只做向量化 argmax", test_step1_argmax_only, 16)
    run_test("步骤 2: argmax + 坐标转换 + argsort", test_step2_with_coordinate_transform, 16)
    run_test("步骤 3: 完整 NMS (含 while 循环)", test_step3_with_nms_loop, 16)
