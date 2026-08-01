# -*- coding: utf-8 -*-
"""
@File    : fastvlm_detector.py
@Time    : 2025/12/16
@Author  : 陈冲 (Refactored)
@Description : 适配 Triton Pipeline 版本的 FastVLM 推理类
@Version : 2.0
"""
import os
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
# 保持原有的工具引用，用于后处理
from api.infer.Triton_model.fastvlm.utils.processing import postprocess
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


def fastvlm_detector(triton_client, service_name, init_data, img, label_to_detect, box_info, conf_thres=-1,
                     iou_thres=-1, conf=None):
    """
    适配后的推理函数：动态读取配置，发送图片 Bytes，接收文本结果
    Args:
        img: OpenCV 格式的 numpy 数组 (BGR)
    """
    pretreatment_start = time.time()

    time_json = {
        "filename": "",
        "model": service_name,
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }

    # -------------------------- 1. 参数准备 & 配置解析 --------------------------
    # 1.2 解析 init_data (动态获取配置)
    if init_data and service_name in init_data:
        cfg = init_data[service_name]

        # --- B. 动态获取 Input Name ---
        input_list = cfg.get("input", [])
        if input_list and len(input_list) > 0:
            # 获取列表第一个 input 的 name 字段
            input_name = input_list[0].get("name", "image_bytes")

        # --- C. 动态获取 Output Name ---
        output_list = cfg.get("output", [])
        if output_list and len(output_list) > 0:
            # 获取列表第一个 output 的 name 字段
            output_name = output_list[0].get("name", "text_output")


    else:
        return [], time_json

    # -------------------------- 2. 图像编码 (OpenCV -> JPG Bytes) --------------------------
    try:
        # ⚠️ 核心提速点：判断 img 的类型
        if isinstance(img, bytes):
            # 如果上游直接传了 Bytes，直接接管，跳过所有 CPU 编解码！
            image_bytes = img
        else:
            # 如果上游传的是 OpenCV 的 Numpy 矩阵（兼容其他老接口），则进行编码
            success, encoded_image = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                logger.error(f"[{service_name}] 图片编码失败")
                return [], time_json
            image_bytes = encoded_image.tobytes()

    except Exception as e:
        logger.error(f"[{service_name}] 预处理(编码)失败: {e}")
        return [], time_json

    pretreatment_end = time.time()

    # -------------------------- 3. Triton 推理 (发送 Bytes) --------------------------
    try:
        # inputs_np = np.array([[image_bytes]], dtype=np.object_)

        # infer_input = grpcclient.InferInput(input_name, [1,1], "BYTES")
        # infer_input.set_data_from_numpy(inputs_np)

        # infer_output = grpcclient.InferRequestedOutput(output_name)
        inputs_np = np.array([[image_bytes]], dtype=np.object_)
        infer_input = grpcclient.InferInput(input_name, [1, 1], "BYTES") # 注意这里用 aio
        infer_input.set_data_from_numpy(inputs_np)
        
        infer_output = grpcclient.InferRequestedOutput(output_name) # 注意这里用 aio

        infer_start = time.time()
        response = triton_client.infer(
            model_name=service_name,
            inputs=[infer_input],
            outputs=[infer_output]
        )
        infer_end = time.time()
        logger.info(f"[{service_name}] 推理耗时: {infer_end - infer_start}")
        result_array = response.as_numpy(output_name)

        if result_array.size > 0:
            raw_text = result_array.flatten()[0]

            if isinstance(raw_text, bytes):
                final_text = raw_text.decode('utf-8')
            elif isinstance(raw_text, np.bytes_):
                final_text = raw_text.decode('utf-8')
            else:
                final_text = str(raw_text)
        else:
            final_text = ""

        logger.info(f"[{service_name}] 服务端返回文本: {final_text}")

    except Exception as e:
        logger.error(f"[{service_name}] Triton 推理失败: {e}")
        return [], time_json

    post_time_end = time.time()

    # -------------------------- 4. 后处理 (Text -> Boxes) --------------------------

    #

    try:
        detected_objects = postprocess(
            final_text,
            1080, 1920,
            1024,
        )
    except Exception as e:
        logger.error(f"[{service_name}] 后处理解析失败: {e}")
        detected_objects = []

    processing_end = time.time()

    # -------------------------- 6. 计时更新 --------------------------
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = post_time_end - pretreatment_end
    time_json["processing_time"] = processing_end - post_time_end
    time_json["showtime"] = processing_end - pretreatment_start

    return detected_objects, time_json
