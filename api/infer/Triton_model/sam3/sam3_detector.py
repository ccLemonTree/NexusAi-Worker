# -*- coding: utf-8 -*-
"""
@File    : sam3_detector.py.py
@Time    : 2025/12/15 16:44
@Author  : 陈冲
@Description : sam3推理
@Version : 1.0
"""

import time
from PIL import Image
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.sam3.utils.processing import *
import numpy as np


def sam3(triton_client, service_name, init_data, img, label_rules, box_info, conf_thres=-1, iou_thres=-1, conf=None):
    pretreatment_start = time.time()  # [时间点1] 预处理开始
    # --- 1. 图像处理 ---
    if isinstance(img, np.ndarray):
        orig_h, orig_w = img.shape[:2]
        img = Image.fromarray(img[..., ::-1])
    else:
        orig_w, orig_h = img.size

    img_resized = img.convert('RGB').resize((1008, 1008))
    # img_data = (np.array(img_resized).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    img_data = np.array(img_resized).astype(np.float32).transpose(2, 0, 1)[None]

    # --- 2. 初始化 time_json ---
    time_json = {
        "filename": "",
        "model": service_name,
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }

    # --- 3. 解析配置 ---
    final_conf_thres = 0.5
    INPUT_CFG = []
    OUTPUT_CFG = []

    if init_data and service_name in init_data:
        cfg = init_data[service_name]
        INPUT_CFG = cfg.get("input", [])
        OUTPUT_CFG = cfg.get("output", [])
        if conf_thres == -1:
            final_conf_thres = cfg.get("conf_thres", 0.5)
        else:
            final_conf_thres = conf_thres
    else:
        if conf_thres != -1: final_conf_thres = conf_thres

    if len(INPUT_CFG) < 2:
        return [], time_json

    # --- 4. 构造 Prompt ---
    label_list_analyse = list(label_rules.keys()) if isinstance(label_rules, dict) else list(label_rules)
    label_list = [i.replace("sam3_", "") for i in label_list_analyse]

    # --- 5. 构造 Triton Inputs 配置 ---
    img_in_name = INPUT_CFG[0]["name"]
    txt_in_name = INPUT_CFG[1]["name"]
    img_type = INPUT_CFG[0]["data_type"].split("_")[-1]
    txt_type = INPUT_CFG[1]["data_type"].split("_")[-1]
    if "STRING" in txt_type:
        txt_type = "BYTES"

    outputs = [grpcclient.InferRequestedOutput(obj["name"]) for obj in OUTPUT_CFG]

    pretreatment_end = time.time()  # [时间点2] 预处理结束

    # --- 6. 分批推理（sam3 最多支持 8 个 label） ---
    MAX_LABELS = 8
    detected_objects = []
    model_ready = triton_client.is_model_ready(service_name)

    for chunk_start in range(0, len(label_list), MAX_LABELS):
        chunk_analyse = label_list_analyse[chunk_start:chunk_start + MAX_LABELS]
        chunk_labels  = label_list[chunk_start:chunk_start + MAX_LABELS]

        TEXT_PROMPTS = np.array([str(x).encode('utf-8') for x in chunk_labels], dtype=object)

        inputs = [
            grpcclient.InferInput(img_in_name, img_data.shape, img_type),
            grpcclient.InferInput(txt_in_name, TEXT_PROMPTS.shape, txt_type)
        ]
        inputs[0].set_data_from_numpy(img_data)
        inputs[1].set_data_from_numpy(TEXT_PROMPTS)

        if not model_ready:
            continue

        results = triton_client.infer(service_name, inputs, outputs=outputs)
        output_dict = {obj["name"]: results.as_numpy(obj["name"]) for obj in OUTPUT_CFG}

        if not output_dict:
            continue

        chunk_rules = {k: label_rules[k] for k in chunk_analyse} if isinstance(label_rules, dict) else chunk_analyse
        detected_objects += postprocess(
            output_dict,
            orig_w,
            orig_h,
            final_conf_thres,
            chunk_analyse,
            label_rules=chunk_rules
        )

    postprocessing_start = time.time()
    postprocessing_end = time.time()

    # --- 8. 时间计算 ---
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start

    return detected_objects, time_json
