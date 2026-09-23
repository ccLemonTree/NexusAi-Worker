# -*- coding: utf-8 -*-
"""
@File    : processing.py
@Time    : 2026/9/7 11:25
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""
"""RF-DETR TensorRT 预处理和后处理。"""
import cv2
import numpy as np

from api.infer.Utils.boundingbox import BoundingBox

def preprocess(img, input_shape):
    """将 OpenCV BGR 图像转换为 RF-DETR 的 NCHW 输入。"""
    height, width = input_shape
    image = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )[:, None, None]
    return np.ascontiguousarray(image[None])


def postprocess(
    dets,
    labels,
    origin_w,
    origin_h,
    label_names,
    min_expand_ratio=0.1,
    max_expand_ratio=1.5,
    ratio_threshold=0.3,
):
    """解码 RF-DETR 结果，并根据目标框大小动态向外扩展。"""
    boxes_cwh = dets[0]
    logits = labels[0, :, :-1]  # 自定义数据集采用末位背景槽。
    scores_all = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all.max(axis=1)

    results = []
    image_area = origin_w * origin_h
    for box, class_id, score in zip(boxes_cwh, class_ids, scores):
        cx, cy, width, height = box
        x1 = float((cx - width / 2) * origin_w)
        y1 = float((cy - height / 2) * origin_h)
        x2 = float((cx + width / 2) * origin_w)
        y2 = float((cy + height / 2) * origin_h)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        box_ratio = (box_w * box_h) / image_area
        if box_ratio < ratio_threshold:
            scale = 1.0 - box_ratio / ratio_threshold
            expand_ratio = min_expand_ratio + (max_expand_ratio - min_expand_ratio) * scale
        else:
            expand_ratio = min_expand_ratio

        expand_w = box_w * expand_ratio / 2
        expand_h = box_h * expand_ratio / 2
        x1 = max(0.0, min(float(origin_w), x1 - expand_w))
        y1 = max(0.0, min(float(origin_h), y1 - expand_h))
        x2 = max(0.0, min(float(origin_w), x2 + expand_w))
        y2 = max(0.0, min(float(origin_h), y2 + expand_h))
        class_id = int(class_id)
        label_name = label_names[class_id] if class_id < len(label_names) else f"ID_{class_id}"
        results.append(BoundingBox(class_id, score, x1, x2, y1, y2, origin_w, origin_h, label_name))
    return results
