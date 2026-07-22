#!/usr/bin/env python3
"""
在导入 numpy 之前设置 BLAS 单线程
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("✅ 已在导入 numpy 前设置 BLAS 单线程环境变量")

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def numpy_heavy_ops():
    """模拟 NMS 中的 numpy 操作"""
    start = time.time()

    # 模拟检测框数据
    prediction = np.random.rand(25200, 85).astype(np.float32)

    # 置信度过滤
    boxes1 = prediction[prediction[:, 4] >= 0.0]

    # argmax（NMS 中最可能的瓶颈）
    zzz = []
    for i in range(len(boxes1)):
        boxe = list(boxes1[i, 0:5])
        boxe.append(int(np.argmax(boxes1[i, 5:])))
        zzz.append(np.array(boxe))
    boxes = np.array(zzz)

    # argsort
    confs = boxes[:, 4]
    boxes = boxes[np.argsort(-confs)]

    # 模拟 NMS 循环
    keep_boxes = []
    while boxes.shape[0] > 0 and len(keep_boxes) < 100:
        keep_boxes.append(boxes[0])
        if boxes.shape[0] == 1:
            break
        box1 = np.expand_dims(boxes[0, :4], 0)
        box2 = boxes[:, :4]
        iou = np.random.rand(boxes.shape[0])
        large_overlap = iou > 0.5
        boxes = boxes[~large_overlap]

    elapsed = time.time() - start
    return True, elapsed

def run_test(concurrent):
    print(f"\n{'='*80}")
    print(f"测试 numpy 操作 (已设置 BLAS 单线程)")
    print(f"并发: {concurrent}")
    print(f"{'='*80}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: numpy_heavy_ops(), range(concurrent)))
    elapsed = time.time() - start_time

    success = sum(1 for r in results if r[0])
    avg = sum(r[1] for r in results) / len(results)

    print(f"总耗时: {elapsed:.3f}s")
    print(f"成功: {success}/{concurrent}")
    print(f"平均单次: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x",
          "✅ 并发" if avg / elapsed > 2 else "❌ 串行")

if __name__ == '__main__':
    run_test(16)
