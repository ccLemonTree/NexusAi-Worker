#!/usr/bin/env python3
"""
测试优化后的 yolov5 NMS 是否变快
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from api.infer.Triton_model.yolov5.utils.processing import non_max_suppression

def test_optimized_nms():
    """测试优化后的 NMS"""
    start = time.time()

    # 模拟 yolov5 输出
    prediction = np.random.rand(25200, 85).astype(np.float32)
    prediction[:, 4] = np.random.rand(25200) * 0.8  # confidence

    boxes = non_max_suppression(
        prediction,
        origin_h=1080,
        origin_w=1920,
        input_w=640,
        input_h=640,
        conf_thres=0.0,
        nms_thres=0.2
    )

    elapsed = time.time() - start
    return elapsed

def run_test(concurrent):
    print(f"\n{'='*80}")
    print(f"测试优化后的 yolov5 NMS")
    print(f"并发: {concurrent}")
    print(f"{'='*80}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: test_optimized_nms(), range(concurrent)))
    elapsed = time.time() - start_time

    avg = sum(results) / len(results)

    print(f"总耗时: {elapsed:.3f}s")
    print(f"平均单次: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x",
          "✅ 并发" if avg / elapsed > 2 else "❌ 串行")

if __name__ == '__main__':
    run_test(16)
