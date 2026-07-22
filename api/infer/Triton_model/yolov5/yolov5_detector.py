import time

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov5.utils.processing import *


def yolov5(triton_client, service_name, init_data, img, label_rules,box_info, conf_thres=-1, iou_thres=-1,conf=None):
    pretreatment_start = time.time()  # 时间测试
    time_json = {
        "filename": "",
        "model": "",
        "Pretreatment": -1,
        "post_time": -1,
        "processing_time": -1,
        "showtime": -1
    }
    time_json["model"] = service_name

    model_version = ""
    # 默认阈值（如果消息和config都没提供，用这个兜底）
    default_conf = 0.5
    default_iou = 0.5
    INPUT_DATA = []
    OUTPUT_DATA = []
    label_names = []
    input_shape = []

    # 1. 从 config.json 读取模型默认阈值
    if not bool(init_data):
        print("模型信息没有读到")
        conf_thres = default_conf
        iou_thres = default_iou
    else:
        for key, value in init_data.items():
            if key == service_name:
                INPUT_DATA = value["input"]
                OUTPUT_DATA = value["output"]
                # config.json 的阈值作为第一优先级默认值
                conf_thres = value.get("conf_thres", default_conf)
                iou_thres = value.get("iou_thres", default_iou)
                model_version = value["model_version"]
                label_names = value["label_names"]

                for input in INPUT_DATA:
                    input_shape.append(input["dims"][-2])
                    input_shape.append(input["dims"][-1])

    # 2. 从 label_rules 提取消息传入的阈值（最高优先级）
    label_thresholds = {}
    if isinstance(label_rules, dict):
        for lbl, rule in label_rules.items():
            if isinstance(rule, dict):
                label_thresholds[lbl] = {
                    "conf": float(rule.get("conf", conf_thres)),
                    "iou": float(rule.get("iou", iou_thres))
                }
            else:
                label_thresholds[lbl] = {"conf": conf_thres, "iou": iou_thres}
    inputs = []
    outputs = []
    output_data = []

    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))
    input_image_buffer = preprocess(img, input_shape)
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    input_image = input_image_buffer.astype(np.float32)
    inputs[0].set_data_from_numpy(input_image)

    pretreatment_end = time.time()  # 时间测试

    if triton_client.is_model_ready(service_name, model_version):
        results = triton_client.infer(model_name=service_name,
                                      inputs=inputs,
                                      outputs=outputs,
                                      model_version=model_version,
                                      client_timeout=30000)
        for output in OUTPUT_DATA:
            results.as_numpy(output["name"])
        for obj in OUTPUT_DATA:
            output_data.append(results.as_numpy(obj["name"]))

    result_to_return =[]
    postprocessing_start = time.time()  # 时间测试
    if len(output_data)==0:
        postprocessing_end = time.time()  # 时间测试
        time_json["Pretreatment"] = pretreatment_end - pretreatment_start
        time_json["post_time"] = postprocessing_start - pretreatment_end
        time_json["processing_time"] = postprocessing_end - postprocessing_start
        return result_to_return, time_json

    # postprocess 不做置信度过滤（传 0.0），由后续的独立阈值过滤
    detected_objects = postprocess(output_data[0], img.shape[1], img.shape[0], input_shape, 0.0,
                                          iou_thres, label_names)
    detects = list(label_rules.keys())
    for i in range(len(detected_objects)):
        box = detected_objects[i]
        label_name = label_names[box.classID]
        if label_name in detects:
            if box.confidence > (label_rules[label_name]['conf'] if isinstance(label_rules,dict) else conf_thres):
                result_to_return.append(box)

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return result_to_return, time_json


