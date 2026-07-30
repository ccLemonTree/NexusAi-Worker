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


def postprocess(output_dict, origin_w, origin_h, conf_thres=0.5, label_names=[], label_rules=None):
    """
    SAM3 后处理 (适配 Return-All 模式)
    Args:
        output_dict: dict, 包含 {'boxes': np.array, 'scores': np.array, 'classes': np.array}
        ... 其他参数不变
    """
    detected_objects = []

    # --- 1. 从字典中提取数据 ---
    # 依靠名字提取，比靠 shape 猜更安全
    # 兼容处理：有些配置可能叫 "classes" 有些可能没改
    boxes = output_dict.get("boxes")
    scores = output_dict.get("scores")
    classes = output_dict.get("classes")  # 这是关键新增项

    # 安全检查
    if boxes is None or scores is None:
        print("[Postprocess Warning] 缺少 boxes 或 scores")
        return detected_objects

    if len(boxes) == 0:
        return detected_objects

    # 如果模型没有返回 classes (旧版模型兼容)，则无法区分多 Prompt，默认全给第0个
    if classes is None:
        classes = np.zeros_like(scores, dtype=np.int32)

    # --- 2. 遍历所有候选框 ---
    # 这里的 boxes 数量可能是 300, 600, 900 等，远大于 label_names 的长度
    for i, (box, score, class_idx) in enumerate(zip(boxes, scores, classes)):

        # --- A. 获取当前框的类别信息 ---
        # 确保 class_idx 是整数
        c_idx = int(class_idx)

        # 防止越界 (比如模型返回了不存在的类别索引)
        if c_idx < 0 or c_idx >= len(label_names):
            continue

        label_name = label_names[c_idx]

        # --- B. 动态获取阈值 (筛选逻辑) ---
        current_thres = conf_thres  # 默认兜底

        # 查找 label_rules
        if isinstance(label_rules, dict) and label_name in label_rules:
            rule = label_rules[label_name]
            # 支持 {'conf': 0.6} 或 直接 0.6 的写法
            if isinstance(rule, dict) and 'conf' in rule:
                current_thres = rule['conf']
            elif isinstance(rule, (float, int)):
                current_thres = rule

        # --- C. 阈值过滤 ---
        if score < float(current_thres):
            continue

        # --- D. 坐标还原 ---
        # 用 min/max 保证 x1<x2、y1<y2，兼容模型以任意角点顺序输出的情况
        raw_x1 = box[0] * origin_w
        raw_y1 = box[1] * origin_h
        raw_x2 = box[2] * origin_w
        raw_y2 = box[3] * origin_h

        x1 = min(raw_x1, raw_x2)
        y1 = min(raw_y1, raw_y2)
        x2 = max(raw_x1, raw_x2)
        y2 = max(raw_y1, raw_y2)

        # 边界截断
        x1 = max(0, min(x1, origin_w))
        y1 = max(0, min(y1, origin_h))
        x2 = max(0, min(x2, origin_w))
        y2 = max(0, min(y2, origin_h))

        try:
            detected_objects.append(
                BoundingBox(
                    c_idx,  # 使用 classes 里取出的真实类别ID
                    float(score),
                    x1, x2, y1, y2,
                    origin_w,
                    origin_h,
                    label_name  # 使用映射后的真实名称
                )
            )
        except Exception as e:
            print(f"[Postprocess Error] {e}")

    return detected_objects
