import time
import io
from PIL import Image
import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.ocr.utils.processing import *


def ocr_det(triton_client, service_name, init_data, img, label_to_detect,box_info, conf_thres=-1, iou_thres=-1,conf=None):
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
    conf_thres = 0.5
    iou_thres = 0.5
    INPUT_DATA = []
    OUTPUT_DATA = []
    label_names = []
    input_shape = []
    if not bool(init_data):
        print("模型信息没有读到")
    else:
        for key, value in init_data.items():
            if key == service_name:
                INPUT_DATA = value["input"]
                OUTPUT_DATA = value["output"]
                if iou_thres == -1:
                    iou_thres = iou_thres
                else:
                    iou_thres = value["iou_thres"]
                if conf_thres == -1:
                    conf_thres = conf_thres
                else:
                    conf_thres = value["conf_thres"]

                model_version = value["model_version"]
                label_names = value["label_names"]

                for input in INPUT_DATA:
                    input_shape.append(input["dims"][-1])
    inputs = []
    outputs = []
    output_data = []
    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    IMAGE_BYTES = buf.getvalue()
    inputs[0].set_data_from_numpy(np.array([IMAGE_BYTES], dtype=object))

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
    flag = True if len(label_to_detect) == 0 else False
    detected_objects = postprocess(output_data[0], img.shape[1], img.shape[0], input_shape, conf_thres,
                                          iou_thres, label_names)
    for i in range(len(detected_objects)):
        box = detected_objects[i]
        if box.confidence > conf_thres if conf is None else conf:
            # box.x1 += box_info.x1
            # box.x2 += box_info.x2
            # box.y1 += box_info.y1
            # box.y2 += box_info.y2
            if flag:
                result_to_return.append(box)
            else:
                if label_names[box.classID] in label_to_detect:
                    result_to_return.append(box)

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return result_to_return, time_json



