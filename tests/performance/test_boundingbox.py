#!/usr/bin/env python3
"""
逐步拆解 postprocess，找出哪个步骤导致串行
"""
import time
import cv2
import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.utils.processing import preprocess
from api.infer.Utils.boundingbox import BoundingBox

def non_max_suppression_only(prediction, origin_h, origin_w, input_w, input_h, conf_thres=0.5, nms_thres=0.4):
    """只做 NMS，不创建 BoundingBox"""
    boxes1 = prediction[prediction[:, 4] >= conf_thres]
    if len(boxes1):
        zzz = []
        for i in range(len(boxes1)):
            boxe = list(boxes1[i, 0:5])
            boxe.append(int(np.argmax(boxes1[i, 5:])))
            zzz.append(np.array(boxe))
        boxes = np.array(zzz)

        # xywh2xyxy inline
        y = np.zeros_like(boxes[:, :4])
        r_w = input_w / origin_w
        r_h = input_h / origin_h
        if r_h > r_w:
            y[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            y[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            y[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y[:, 3] = boxes[:, 1] + boxes[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = boxes[:, 0] - boxes[:, 2] / 2 - (input_w - r_h * origin_w) / 2
            y[:, 2] = boxes[:, 0] + boxes[:, 2] / 2 - (input_w - r_h * origin_w) / 2
            y[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            y[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            y /= r_h
        boxes[:, :4] = y

        # clip
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)

        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]

        # NMS loop
        keep_boxes = []
        while boxes.shape[0]:
            # bbox_iou inline
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
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    else:
        boxes = []
    return boxes

def test_1_nms_only(img, triton_url):
    """测试 1: 推理 + NMS（不创建 BoundingBox 对象）"""
    start = time.time()
    client = grpcclient.InferenceServerClient(url=triton_url, channel_args=[
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ])

    try:
        input_shape = [640, 640]
        input_image_buffer = preprocess(img, input_shape)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0).astype(np.float32)

        inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_image_buffer)
        outputs = [grpcclient.InferRequestedOutput("output")]

        if client.is_model_ready("yolov5_persondog", model_version="1"):
            results = client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs, model_version="1", client_timeout=30000)
            output_data = results.as_numpy("output")

            # 只做 NMS
            nc = output_data.shape[2] - 5
            pred = np.reshape(output_data[0:], (-1, nc+5))
            boxes = non_max_suppression_only(pred, img.shape[0], img.shape[1], 640, 640, 0.0, 0.2)

        elapsed = time.time() - start
        return True, elapsed
    except Exception as e:
        return False, time.time() - start, str(e)
    finally:
        client.close()

def test_2_with_boundingbox(img, triton_url):
    """测试 2: 推理 + NMS + 创建 BoundingBox 对象"""
    start = time.time()
    client = grpcclient.InferenceServerClient(url=triton_url, channel_args=[
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ])

    try:
        input_shape = [640, 640]
        input_image_buffer = preprocess(img, input_shape)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0).astype(np.float32)

        inputs = [grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")]
        inputs[0].set_data_from_numpy(input_image_buffer)
        outputs = [grpcclient.InferRequestedOutput("output")]

        if client.is_model_ready("yolov5_persondog", model_version="1"):
            results = client.infer(model_name="yolov5_persondog", inputs=inputs, outputs=outputs, model_version="1", client_timeout=30000)
            output_data = results.as_numpy("output")

            # NMS
            nc = output_data.shape[2] - 5
            pred = np.reshape(output_data[0:], (-1, nc+5))
            boxes = non_max_suppression_only(pred, img.shape[0], img.shape[1], 640, 640, 0.0, 0.2)

            # 创建 BoundingBox 对象
            label_names = ["personnew", "dognew", "catnew"]
            detected_objects = []
            result_boxes = boxes[:, :4] if len(boxes) else np.array([])
            result_scores = boxes[:, 4] if len(boxes) else np.array([])
            result_classid = boxes[:, 5].astype(np.int_) if len(boxes) else np.array([])

            for box, score, label in zip(result_boxes, result_scores, result_classid):
                detected_objects.append(BoundingBox(
                    label, score, box[0], box[2], box[1], box[3],
                    img.shape[1], img.shape[0], label_names[label]
                ))

        elapsed = time.time() - start
        return True, elapsed
    except Exception as e:
        return False, time.time() - start, str(e)
    finally:
        client.close()

def run_test(name, func, img, url, concurrent):
    print(f"\n{'='*80}")
    print(name)
    print(f"{'='*80}")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: func(img, url), range(concurrent)))
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r[0])
    avg = sum(r[1] for r in results) / len(results)
    print(f"总耗时: {elapsed:.3f}s  QPS: {concurrent / elapsed:.2f}")
    print(f"成功: {success}/{concurrent}, 平均: {avg:.3f}s")
    print(f"并发倍数: {avg / elapsed:.2f}x", "✅ 并发" if avg / elapsed > 2 else "❌ 串行")
    if not results[0][0]:
        print(f"❌ 错误: {results[0][2]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--concurrent', type=int, default=16)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取: {args.image}")
        return

    print(f"图片: {img.shape}, 并发: {args.concurrent}")
    url = os.getenv("TRITON_SERVER", "10.120.0.66:40008")

    run_test("测试 1: 推理 + NMS (不创建 BoundingBox)", test_1_nms_only, img, url, args.concurrent)
    run_test("测试 2: 推理 + NMS + 创建 BoundingBox", test_2_with_boundingbox, img, url, args.concurrent)

if __name__ == '__main__':
    main()
