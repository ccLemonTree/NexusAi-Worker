# -*- coding: utf-8 -*-
"""
@File    : cangqiong.py
@Time    : 2026/9/21 11:14
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""

import time
import base64
import cv2
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


def img_to_b64(img: np.ndarray, max_side=1024,flag = False):
    """
    和call_qwen内部逻辑对齐：opencv BGR图等比缩放，输出jpeg base64字符串
    :param img: opencv BGR numpy array
    :param max_side: 长边最大尺寸
    :return: base64字符串
    """
    h, w = img.shape[:2]
    if flag:
        success, img_encoded = cv2.imencode(".png", img)
        if not success:
            raise RuntimeError("image encode jpeg failed")
        img_bytes = img_encoded.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')

    scale = max_side / max(h, w)
    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h))
    success, img_encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("image encode jpeg failed")
    img_bytes = img_encoded.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')


def cangqiong_detector(triton_client, service_name, init_data, img, label_to_detect, box_info, conf_thres=-1,
                       iou_thres=-1, conf=None):
    """
    适配后的推理函数：支持【原图 + 多个crop子图】多图输入，对齐call_qwen逻辑
    完整匹配OpenAI API的messages格式（system + user + 多张image）

    Args:
        triton_client: Triton gRPC 客户端
        service_name: 服务/模型名称
        init_data: 配置数据
        img: 保留旧接口兼容，可传None；多图模式使用original_img + crops
        label_to_detect: 要检测的标签
        box_info: 框信息
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        conf: 额外配置
        original_img: 原图 opencv BGR array，多图模式必填
        crops: list[np.ndarray]，各个crop裁剪图，可为空列表
    Returns:
        detected_objects: 检测结果列表
        time_json: 时间统计字典
    """
    original_img, crops = img[0], img[1:]
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
    system_message = cfg.get("system_message", )
    prompt = cfg.get("prompt_text", "")
    label = cfg.get("label", [])
    model_name = cfg.get("model_name", service_name)
    print(cfg)
    # 多图处理逻辑：对齐call_qwen：content = [text,原图, text, crop1, text, crop2...]
    # 构建Qwen聊天模板内的user侧文本部分；image占位符 <|vision_start|><|image_pad|><|vision_end|> 一一对应每张图
    user_text_parts = []
    vision_token_list = []

    if original_img is not None:
        user_text_parts.append(prompt)
        vision_token_list.append("<|vision_start|><|image_pad|><|vision_end|>")
        # 依次追加各个crop
        if crops is not None and len(crops) > 0:
            for idx, _ in enumerate(crops):
                user_text_parts.append(f"--- 帧 {idx + 1} ---")
                vision_token_list.append("<|vision_start|><|image_pad|><|vision_end|>")

    user_prompt = "\n".join(user_text_parts)
    # 拼接完整prompt：system + user(文本 + N组image占位) + assistant引导
    # 注意: Qwen3 系列默认开启 thinking, 仅靠 prompt 文本"禁止深度思考"无效,
    # 必须在 assistant 后插入闭合的空 think 块 <think>\n\n</think>\n\n 才能强制跳过思考阶段
    full_prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n"
        + "".join(vision_token_list) + "\n"
        + f"{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    # -------------------------- 2. 图像编码 (OpenCV -> JPG -> Base64) 多张图片 --------------------------
    image_b64_list = []
    try:
        if original_img is not None:
            # 原图max_side=1024
            b64_ori = img_to_b64(original_img, flag=False)
            image_b64_list.append(b64_ori)
            # crop图max_side=512
            if crops is not None:
                for crop_img in crops:
                    b64_crop = img_to_b64(crop_img, flag=True)
                    image_b64_list.append(b64_crop)

        # 兼容旧单图逻辑：当original_img=None，使用传入img
        if len(image_b64_list) == 0 and img is not None:
            h, w, c = img.shape
            scale = 1024 / w
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            img = cv2.resize(img, (new_w, new_h))
            success, img_encoded = cv2.imencode(".jpeg", img)
            if not success:
                logger.error(f"[{service_name},{img.shape}] 图片编码失败")
                return [], time_json
            img_bytes = img_encoded.tobytes()
            image_b64_list.append(base64.b64encode(img_bytes).decode('utf-8'))

        logger.info(f"[{service_name}] 图片编码成功，图片数量:{len(image_b64_list)}，单张base64样例长度: {len(image_b64_list[0]) if image_b64_list else 0}")

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

        # 1. 文本输入（完整的对话格式，包含system和user，带N个vision占位符）
        text_data = np.array([full_prompt.encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(text_data)

        # 2. image输入：BYTES数组，shape [num_images]，每个元素为单张图片base64字符串
        # Triton vLLM backend 要求 image 输入为 1D：[-1]，即 [num_images]
        image_array = np.array([x.encode("utf-8") for x in image_b64_list], dtype=np.object_)
        inputs.append(grpcclient.InferInput("image", [len(image_b64_list)], "BYTES"))
        inputs[-1].set_data_from_numpy(image_array)

        # 3. stream - 设为False（让backend一次性生成）
        stream_data = np.array([False], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        # 4. sampling_parameters（对齐 vLLM Triton backend 的 SamplingParams）
        # 注意：vLLM 的 SamplingParams 不支持 chat_template_kwargs（那是 chat template 的参数，不是 SamplingParams 字段）
        # TritonSamplingParams.from_dict 遇到未知字段会抛异常并返回 None，导致后续 'NoneType' object has no attribute 'lora_name'
        sampling_params = {
            "temperature": str(cfg.get("temperature", 0.1)),
            "top_p": str(cfg.get("top_p", 0.1)),
            "max_tokens": str(cfg.get("max_tokens",100)),
            "top_k": str(cfg.get("top_k", 20))
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
        import re as _re
        full_result = _re.sub(r"<think>.*?</think>\s*", "", full_result, flags=_re.DOTALL)
        full_result = full_result.replace('\n\n', "").replace("```","")
        if full_result:
            logger.info(f"[{service_name}] 推理完成，输出: {full_result[:100]}...")

            # -------------------------- 4. 后处理：解析结果 --------------------------
            # 根据label配置解析检测结果，保持原有BoundingBox输出格式不变
            labels = list(label_to_detect.keys())[0]
            # 获取原图宽高：优先original_img，回退到老img
            ref_img = original_img if original_img is not None else img
            h_ref, w_ref = ref_img.shape[:2]
            detected_objects.append(
                BoundingBox(0, 0.99, 0, 1, 0, 1, w_ref, h_ref, labels, {"analyse_desc": full_result})
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
