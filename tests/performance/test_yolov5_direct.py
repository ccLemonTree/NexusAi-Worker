#!/usr/bin/env python3
"""
直接调用 yolov5() 函数，完全跳过 tritonServer 单例，测试是否还串行
"""
import time
import cv2
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.yolov5_detector import yolov5

def load_config():
    """加载 yolov5_persondog 的配置"""
    config_path = "api/infer/model_data/yolov5_persondog/config.json"
    with open(config_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    init_data = {
        json_data["name"]: {
            "name": json_data["name"],
            "label_names": json_data["classes"],
            "input": json_data["input"],
            "output": json_data["output"],
            "model_version": str(json_data["model_version"][-1]),
            "iou_thres": json_data["iou_thres"],
            "conf_thres": json_data["conf_thres"]
        }
    }
    return init_data

def single_inference(img, init_data, triton_url):
    """单次推理：独立创建 client"""
    start = time.time()

    # 每次创建独立的 gRPC 客户端
    client = grpcclient.InferenceServerClient(
        url=triton_url,
        channel_args=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    )

    try:
        # 直接调用 yolov5 函数，不经过 tritonServer 单例
        result, time_json = yolov5(
            triton_client=client,
            service_name="yolov5_persondog",
            init_data=init_data,
            img=img,
            label_rules={"personnew": {"conf": 0.5, "iou": 0.5}},
            box_info=None
        )
        elapsed = time.time() - start
        return True, elapsed, len(result)
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)
    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='测试图片路径')
    parser.add_argument('--concurrent', type=int, default=16, help='并发数')
    args = parser.parse_args()

    # 加载图片
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取图片: {args.image}")
        return

    print(f"图片: {img.shape}, 并发: {args.concurrent}")

    # 加载配置
    init_data = load_config()
    triton_url = os.getenv("TRITON_SERVER", "10.120.0.66:40008")

    # 并发测试
    print(f"\n{'='*80}")
    print(f"直接调用 yolov5() 函数 (完全跳过 tritonServer 单例)")
    print(f"{'='*80}")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = [
            executor.submit(single_inference, img, init_data, triton_url)
            for _ in range(args.concurrent)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start_time

    # 统计
    success = sum(1 for r in results if r[0])
    avg_time = sum(r[1] for r in results) / len(results)

    print(f"总耗时: {elapsed:.3f}s  QPS: {args.concurrent / elapsed:.2f}")
    print(f"成功: {success}/{args.concurrent}")
    print(f"平均单次: {avg_time:.3f}s")
    print(f"并发倍数: {avg_time / elapsed:.2f}x",
          "✅ 并发" if avg_time / elapsed > 2 else "❌ 串行")

    if not results[0][0]:
        print(f"\n❌ 错误: {results[0][2]}")

if __name__ == '__main__':
    main()
