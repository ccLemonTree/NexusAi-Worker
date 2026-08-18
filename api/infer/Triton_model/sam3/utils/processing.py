# -*- coding: utf-8 -*-
"""
@File    : processing.py.py
@Time    : 2025/12/15 17:58
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""
import numpy as np

from api.infer.Utils.boundingbox import BoundingBox


def nms(boxes, scores, iou_threshold=0.45):
    """
    boxes: np.ndarray [N,4] x1,y1,x2,y2
    scores: np.ndarray [N]
    return:保留的索引
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(output_dict, origin_w, origin_h, conf_thres=0.5, label_names=[], label_rules=None):
    """
    SAM3 后处理 (适配 Return-All 模式)
    注意：模型输出boxes格式 [cx, cy, w, h] 归一化0~1
    """
    boxes = output_dict.get("boxes")
    scores = output_dict.get("scores")
    classes = output_dict.get("classes")

    if boxes is None or scores is None:
        print("[Postprocess Warning] 缺少 boxes 或 scores")
        return []

    if len(boxes) == 0:
        return []

    if classes is None:
        classes = np.zeros_like(scores, dtype=np.float32)

    # 先收集原始解析后的框、分数、类别，未生成BoundingBox
    raw_box_list = []
    raw_score_list = []
    raw_cls_idx_list = []

    for box, score, class_idx in zip(boxes, scores, classes):
        c_idx = int(round(float(class_idx)))
        if c_idx < 0 or c_idx >= len(label_names):
            continue

        label_name = label_names[c_idx]

        current_thres = conf_thres
        if isinstance(label_rules, dict) and label_name in label_rules:
            rule = label_rules[label_name]
            if isinstance(rule, dict) and 'conf' in rule:
                current_thres = rule['conf']
            elif isinstance(rule, (float, int)):
                current_thres = rule

        if score < float(current_thres):
            continue

        # cx cy w h 转 xmin ymin xmax ymax
        cx_norm = box[0]
        cy_norm = box[1]
        bw_norm = box[2]
        bh_norm = box[3]

        xmin_norm = cx_norm - bw_norm / 2.0
        ymin_norm = cy_norm - bh_norm / 2.0
        xmax_norm = cx_norm + bw_norm / 2.0
        ymax_norm = cy_norm + bh_norm / 2.0

        x1 = xmin_norm * origin_w
        y1 = ymin_norm * origin_h
        x2 = xmax_norm * origin_w
        y2 = ymax_norm * origin_h

        # 边界截断
        x1 = max(0.0, min(x1, origin_w))
        y1 = max(0.0, min(y1, origin_h))
        x2 = max(0.0, min(x2, origin_w))
        y2 = max(0.0, min(y2, origin_h))

        raw_box_list.append([x1, x2, y1, y2])
        raw_score_list.append(float(score))
        raw_cls_idx_list.append(c_idx)

    if len(raw_box_list) == 0:
        return []

    # NMS过滤
    box_arr = np.array(raw_box_list, dtype=np.float32)
    score_arr = np.array(raw_score_list, dtype=np.float32)
    keep_idx = nms(box_arr, score_arr, iou_threshold=0.45)

    detected_objects = []
    for idx in keep_idx:
        x1, x2, y1, y2 = raw_box_list[idx]
        s = raw_score_list[idx]
        c_idx = raw_cls_idx_list[idx]
        label_name = label_names[c_idx]
        try:
            detected_objects.append(
                BoundingBox(
                    c_idx,
                    s,
                    x1, x2, y1, y2,
                    origin_w,
                    origin_h,
                    label_name
                )
            )
        except Exception as e:
            print(f"[Postprocess Error] {e}")

    return detected_objects
