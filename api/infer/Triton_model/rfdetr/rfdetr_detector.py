# -*- coding: utf-8 -*-
"""
@File    : rfdetr_detector.py
@Time    : 2026/9/7 11:24
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""
"""RF-DETR 检测接口。"""
import time

import numpy as np
import tritonclient.grpc as grpcclient

from api.infer.Triton_model.rfdetr.utils.processing import postprocess, preprocess


def rfdetr(triton_client, service_name, init_data, img, label_to_detect, box_info, conf_thres=-1):
    start = time.time()
    time_json = {"filename": "", "model": service_name, "Pretreatment": -1, "post_time": -1, "processing_time": -1, "showtime": -1}
    model_config = init_data[service_name]
    conf_thres = model_config["conf_thres"] if conf_thres == -1 else conf_thres
    input_shape = model_config["input"][0]["dims"][-2:]
    model_version = model_config["model_version"]
    inputs = [grpcclient.InferInput(item["name"], item["dims"], item["data_type"].split("_")[-1]) for item in model_config["input"]]
    outputs = [grpcclient.InferRequestedOutput(item["name"]) for item in model_config["output"]]
    inputs[0].set_data_from_numpy(preprocess(img, input_shape).astype(np.float32))
    prepared = time.time()

    if not triton_client.is_model_ready(service_name, model_version):
        time_json.update(Pretreatment=prepared - start, post_time=0, processing_time=0)
        return [], time_json

    result = triton_client.infer(model_name=service_name, inputs=inputs, outputs=outputs, model_version=model_version, client_timeout=30000)
    inferred = time.time()
    boxes = postprocess(result.as_numpy("dets"), result.as_numpy("labels"), img.shape[1], img.shape[0], model_config["label_names"])
    thresholds = {
        name: float(rule.get("conf", conf_thres)) if isinstance(rule, dict) else conf_thres
        for name, rule in label_to_detect.items()
    } if isinstance(label_to_detect, dict) else {}
    boxes = [box for box in boxes if box.confidence > thresholds.get(box.classname, conf_thres) and (not label_to_detect or box.classname in label_to_detect)]
    finished = time.time()
    time_json.update(Pretreatment=prepared - start, post_time=inferred - prepared, processing_time=finished - inferred)
    return boxes, time_json
