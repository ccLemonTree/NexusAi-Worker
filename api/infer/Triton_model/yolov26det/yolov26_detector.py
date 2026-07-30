import time

import cv2
import tritonclient.grpc as grpcclient
from api.infer.Triton_model.yolov26det.utils.processing import *
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


def yolov26det(triton_client, service_name, init_data, img, label_to_detect,box_info, conf_thres=-1, iou_thres=-1,conf=None):
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

    # 2. 从 label_to_detect 提取消息传入的阈值（最高优先级）
    # label_to_detect 格式: {"26_fire": {"conf": 0.62, "iou": 0.2}, "26_smoke": {"conf": 0.65, "iou": 0.3}, ...}
    # 保留每个标签独立的阈值字典，供后处理使用
    label_thresholds = {}
    if isinstance(label_to_detect, dict):
        for lbl, rule in label_to_detect.items():
            if isinstance(rule, dict):
                label_thresholds[lbl] = {
                    "conf": float(rule.get("conf", conf_thres)),
                    "iou": float(rule.get("iou", iou_thres))
                }
            else:
                # 兼容旧格式 label_to_detect 是 list 的情况
                label_thresholds[lbl] = {"conf": conf_thres, "iou": iou_thres}
    inputs = []
    outputs = []
    output_data = []
    # cv2.imshow("1",img)
    # cv2.waitKey(0)
    for input in INPUT_DATA:
        type_data = input["data_type"].split("_")
        inputs.append(grpcclient.InferInput(input["name"], input["dims"], type_data[-1]))
    for OutputName in OUTPUT_DATA:
        outputs.append(grpcclient.InferRequestedOutput(OutputName["name"]))
    # input_image_buffer = preprocess(img, input_shape)
    # input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    input_image_buffer, meta = preprocess(img, input_shape)
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
            arr = results.as_numpy(obj["name"])
            output_data.append(arr)
            # 原始输出快照：帮助判断Triton返回值是否异常
            if len(arr.shape) == 3:
                scores_col = arr[0, :, 4]  # [1,300,6] 第4列 = score
                logger.debug(
                    f"[yolov26det] {service_name} raw output shape={arr.shape} "
                    f"score_col max={float(scores_col.max()):.4f} "
                    f"mean={float(scores_col.mean()):.4f} "
                    f"nonzero={int((scores_col > 0).sum())}"
                )
    else:
        logger.warning(f"[yolov26det] 模型未就绪，跳过推理: service={service_name!r} model_version={model_version!r}")

    result_to_return =[]
    postprocessing_start = time.time()  # 时间测试
    if len(output_data)==0:
        postprocessing_end = time.time()  # 时间测试
        time_json["Pretreatment"] = pretreatment_end - pretreatment_start
        time_json["post_time"] = postprocessing_start - pretreatment_end
        time_json["processing_time"] = postprocessing_end - postprocessing_start
        return result_to_return, time_json

    flag = True if len(label_to_detect) == 0 else False

    # postprocess 不做置信度过滤（传 0.0），所有框都通过，由后续的独立阈值过滤
    detected_objects = postprocess(output_data[0], img.shape[1], img.shape[0], input_shape, 0.0,
                                          iou_thres, label_names)

    logger.info(
        f"[yolov26det] {service_name} 原始检测框={len(detected_objects)}  "
        f"label_thresholds={label_thresholds}  flag={flag}"
    )

    # 过滤：每个标签使用独立的阈值
    for i in range(len(detected_objects)):
        box = detected_objects[i]
        label_name = label_names[box.classID]

        # 获取该标签的独立阈值
        if label_name in label_thresholds:
            label_conf = label_thresholds[label_name]["conf"]
        else:
            label_conf = conf_thres  # 降级到全局阈值

        # 置信度过滤
        if box.confidence > label_conf:
            if flag:
                # 无标签过滤，返回所有
                result_to_return.append(box)
            else:
                # 标签过滤
                if label_name in label_to_detect:
                    result_to_return.append(box)

    if not result_to_return:
        # 打印置信度最高的前5个框，辅助判断是阈值问题还是模型无检测
        top5 = sorted(detected_objects, key=lambda b: b.confidence, reverse=True)[:5]
        top5_info = [(label_names[b.classID], round(b.confidence, 4)) for b in top5]
        logger.info(f"[yolov26det] {service_name} 过滤后无结果，置信度Top5: {top5_info}")

    postprocessing_end = time.time()  # 时间测试
    time_json["Pretreatment"] = pretreatment_end - pretreatment_start
    time_json["post_time"] = postprocessing_start - pretreatment_end
    time_json["processing_time"] = postprocessing_end - postprocessing_start
    return result_to_return, time_json



