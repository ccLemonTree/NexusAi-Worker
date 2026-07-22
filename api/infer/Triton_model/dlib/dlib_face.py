import time
import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.dlib.utils.preprocess import preprocess_image,postprocess
import numpy as np
def dlib_face_vector(triton_client, service_name, init_data, img, label_rules,box_info, conf_thres=-1, iou_thres=-1,conf=None):
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

    # 从 config.json 读取模型默认阈值
    if not bool(init_data):
        print("模型信息没有读到")
        conf_thres = default_conf
        iou_thres = default_iou
    else:
        for key, value in init_data.items():
            if key == service_name:
                INPUT_DATA = value["input"]
                OUTPUT_DATA = value["output"]
                # config.json 的阈值作为默认值
                conf_thres = value.get("conf_thres", default_conf)
                iou_thres = value.get("iou_thres", default_iou)
                model_version = value["model_version"]
                label_names = value["label_names"]

                for input in INPUT_DATA:
                    input_shape.append(input["dims"][-2])
                    input_shape.append(input["dims"][-1])
    inputs = []
    outputs = []
    output_data = []

    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))


    input_image_buffer = preprocess_image(img)
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

    detected_objects = postprocess(output_data[0], img.shape[1], img.shape[0], input_shape, conf_thres,
                                          iou_thres, label_names)

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return detected_objects, time_json


