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
import io
import base64
import cv2
import numpy as np
from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import json
import queue

from api.infer.Utils.boundingbox import BoundingBox
# 保持原有的工具引用，用于后处理
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


def _is_final_response(result):
    """检查是否是最终响应"""
    try:
        response = result.get_response()
        for param in response.parameters:
            if param == "triton_final_response":
                return response.parameters[param].bool_param
        return False
    except:
        return False


def cangqiong_detector(triton_client, service_name, init_data, img, label_to_detect, box_info, conf_thres=-1,
                       iou_thres=-1, conf=None):
    """
    适配后的推理函数：动态读取配置，发送图片 Bytes，接收文本结果
    完整匹配OpenAI API的messages格式（system + user + image）

    Args:
        triton_client: Triton gRPC 客户端
        service_name: 服务/模型名称
        init_data: 配置数据
        img: OpenCV 格式的 numpy 数组 (BGR)
        label_to_detect: 要检测的标签
        box_info: 框信息
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        conf: 额外配置
    Returns:
        detected_objects: 检测结果列表
        time_json: 时间统计字典
    """
    pretreatment_start = time.time()
    detected_objects = []
    time_json = {
        "filename": "",
        "model": service_name,
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }

    # -------------------------- 1. 参数准备 & 配置解析 --------------------------
    if not init_data or service_name not in init_data:
        logger.error(f"[{service_name}] 配置不存在")
        return [], time_json

    cfg = init_data[service_name]

    # 获取配置（对应OpenAI API的messages）
    system_message = cfg.get("system_message", "描述红色框里的内容，并结合周边图片信息对内容进行研判和补充。")
    prompt = cfg.get("prompt", "描述红色框里的内容，并结合周边图片信息对内容进行研判和补充。")
    label = cfg.get("label", [])
    model_name = cfg.get("model_name", service_name)

    # 构建完整提示词（匹配OpenAI API的messages格式）
    # 添加 <|im_start|>assistant 让模型直接开始生成，避免输出 "assistant" 前缀
    user_prompt = f"{prompt}"

    full_prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"  # 添加assistant开始标记，模板会自动添加 <think>\n\n</think>
    )

    # -------------------------- 2. 图像编码 (OpenCV -> JPG -> Base64) --------------------------
    try:
        # 编码为JPEG
        h, w, c = img.shape
        scale = 1024 / w
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        # 等比缩放
        img = cv2.resize(img, (new_w, new_h))
        success, img_encoded = cv2.imencode(".jpeg", img)
        if not success:
            logger.error(f"[{service_name}] 图片编码失败")
            return [], time_json

        # 转为base64（对应OpenAI API中的 data:image/jpeg;base64,{base64_data}）
        img_bytes = img_encoded.tobytes()
        image_base64 = base64.b64encode(img_bytes).decode('utf-8')

        logger.info(f"[{service_name}] 图片编码成功，Base64长度: {len(image_base64)}")

    except Exception as e:
        logger.error(f"[{service_name}] 预处理(编码)失败: {e}")
        return [], time_json

    pretreatment_end = time.time()

    # -------------------------- 3. Triton 推理 (使用流式API但表现为非流式) --------------------------
    try:
        # 用队列收集响应
        response_queue = queue.Queue()

        def callback(result, error):
            """流式响应回调函数"""
            if error:
                response_queue.put(('error', error))
            else:
                response_queue.put(('result', result))

        # 构建输入
        inputs = []

        # 1. 文本输入（完整的对话格式，包含system和user）
        text_data = np.array([full_prompt.encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(text_data)

        # 2. 图片输入（纯base64，对应image_url中的base64部分）
        image_array = np.array([image_base64.encode('utf-8')], dtype=np.object_)
        inputs.append(grpcclient.InferInput("image", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(image_array)

        # 3. stream - 设为False（让backend一次性生成）
        stream_data = np.array([False], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        # 4. sampling_parameters（对应OpenAI API的temperature和top_p）
        sampling_params = {
            "temperature": str(cfg.get("temperature", 0.1)),
            "top_p": str(cfg.get("top_p", 0.1)),
            "max_tokens": str(cfg.get("max_tokens", 100))
        }
        params_data = np.array([json.dumps(sampling_params).encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(params_data)

        # 5. exclude_input_in_output
        exclude_data = np.array([True], dtype=bool)
        inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(exclude_data)

        # 输出
        outputs = [grpcclient.InferRequestedOutput("text_output")]

        logger.info(f"[{service_name}] 开始推理...")

        # 必须使用流式API（decoupled policy要求）
        triton_client.start_stream(callback=callback)

        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=f"{service_name}_request",
            model_version="1"
        )

        result_text = []
        inference_start = time.time()

        # 收集所有响应
        while True:
            try:
                item = response_queue.get(timeout=120.0)

                if item[0] == 'error':
                    error = item[1]
                    logger.error(f"[{service_name}] 推理错误: {error}")
                    triton_client.stop_stream()
                    return [], time_json

                result = item[1]

                # 提取文本输出
                try:
                    output = result.as_numpy("text_output")
                    for text_item in output:
                        text = text_item.decode("utf-8")
                        result_text.append(text)
                except Exception:
                    pass

                # 检查是否是最终响应
                if _is_final_response(result):
                    break

            except queue.Empty:
                logger.error(f"[{service_name}] 推理超时：120秒内未收到响应")
                triton_client.stop_stream()
                return [], time_json

        triton_client.stop_stream()
        inference_end = time.time()

        full_result = "".join(result_text)
        full_result = full_result.replace('<think>\n\n',"")
        if full_result:
            logger.info(f"[{service_name}] 推理完成，输出: {full_result[:100]}...")

            # -------------------------- 4. 后处理：解析结果 --------------------------
            # 根据label配置解析检测结果
            labels = list(label_to_detect.keys())[0]
            detected_objects.append(
                BoundingBox(0, 0.99, 0, 1, 0, 1, w, h, labels, {"analyse_desc": full_result})
            )

        else:
            logger.warning(f"[{service_name}] 模型无输出")

    except Exception as e:
        logger.error(f"[{service_name}] 推理异常: {e}")
        import traceback
        traceback.print_exc()
        return [], time_json

    processing_end = time.time()

    # -------------------------- 5. 计时更新 --------------------------
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = inference_end - pretreatment_end
    time_json["processing_time"] = processing_end - inference_end
    time_json["showtime"] = processing_end - pretreatment_start

    return detected_objects, time_json
