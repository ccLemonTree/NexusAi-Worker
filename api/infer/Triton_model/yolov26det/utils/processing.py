from api.infer.Utils.boundingbox import BoundingBox
import cv2
import numpy as np
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

def preprocess(img: np.ndarray, input_shape=(640, 640), stride=32):
    """
    专为 TensorRT 部署设计的纯 OpenCV/NumPy 预处理 (对齐 Ultralytics 官方 LetterBox)
    
    Args:
        img: OpenCV 读取的图片，BGR 格式，形状为 (H, W, 3)
        input_shape: 模型要求的输入尺寸 (height, width)，通常为 (640, 640)
        stride: 模型步长，用于确保填充后的尺寸可以被整除
        
    Returns:
        input_tensor: 适合 TensorRT 输入的 NumPy 数组，形状为 (1, 3, H, W)，float32 类型
        meta: 包含缩放比例和 padding 信息的字典，用于后处理坐标还原
    """
    img_h, img_w = img.shape[:2]
    new_h, new_w = input_shape[0], input_shape[1]

    # 1. 计算缩放比例 (选择较小的比例以保证图片完全容纳在目标尺寸内)
    r = min(new_w / img_w, new_h / img_h)
    
    # 2. 计算缩放后的实际大小 (四舍五入)
    unpad_w = int(round(img_w * r))
    unpad_h = int(round(img_h * r))

    # 3. 计算需要填充的四周 padding
    dw = new_w - unpad_w
    dh = new_h - unpad_h
    
    dw /= 2  # 居中平分
    dh /= 2

    # 4. 执行图片缩放
    if (img_w, img_h) != (unpad_w, unpad_h):
        resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img

    # 5. 复刻官方的像素对齐：计算上下左右具体的填充像素
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 填充四周，Ultralytics 官方默认使用 114 灰色
    padded_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 6. HWC -> CHW, BGR -> RGB 核心标准转换
    # 提示：Windows/Linux 下，下述切片和 transpose 会导致内存不连续
    rgb_img = padded_img[:, :, ::-1]                   # BGR 转换为 RGB
    chw_img = rgb_img.transpose((2, 0, 1))             # (H, W, 3) -> (3, H, W)
    
    # 7. 归一化并强制转换为内存连续数组 (连续内存对 TensorRT 至关重要)
    chw_img = np.ascontiguousarray(chw_img, dtype=np.float32)
    chw_img /= 255.0                                   # 0-255 -> 0.0-1.0

    # 8. 增加 Batch 维度 (3, H, W) -> (1, 3, H, W)
    input_tensor = np.expand_dims(chw_img, axis=0)

    # 9. 记录元数据，用于后续对 [1, 300, 6] 的输出进行坐标还原
    meta = {
        "scale": r, 
        "pad": (left, top), 
        "orig_shape": (img_h, img_w)
    }

    return input_tensor, meta


def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, label_names=None):
    """
    专为 YOLO26 TensorRT [1, 300, 6] 格式适配的后处理方法
    (已移除冗余的 NMS 逻辑，完美对接你的 Letterbox 逆映射)
    """
    if label_names is None:
        label_names = []

    # 1. 调整输出维度 [1, 300, 6] -> [300, 6]
    if len(output.shape) == 3:
        pred = output[0]  # 直接取第一个 batch，shape 变为 (300, 6)
    else:
        pred = output

    detected_objects = []
    if len(pred) == 0:
        return detected_objects

    # 2. 计算 Letterbox 的逆映射参数（保持你原有的逻辑）
    # gain = 旧图 / 新图 的缩放比例 (取宽高中较小的那个，保持等比例)
    gain = min(input_shape[1] / origin_w, input_shape[0] / origin_h) 
    
    # pad_x, pad_y 是在 640x640 画布上单侧填充的灰边像素数
    pad_x = (input_shape[1] - origin_w * gain) / 2
    pad_y = (input_shape[0] - origin_h * gain) / 2

    # 3. 遍历 YOLO26 直接输出的 300 个候选框
    for b in pred:
        # YOLO26 默认输出格式通常为: [x1, y1, x2, y2, score, cls_id]
        x1, y1, x2, y2, score, cls_id = b

        # 注意：置信度过滤已移至 detector 层（支持每个标签独立阈值）
        cls_id = int(cls_id)

        # 5. 减去灰边，并除以缩放系数，还原到原图物理尺寸
        x1_scaled = (x1 - pad_x) / gain
        y1_scaled = (y1 - pad_y) / gain
        x2_scaled = (x2 - pad_x) / gain
        y2_scaled = (y2 - pad_y) / gain

        # 6. 裁剪越界坐标（防止预测框超出原图边界）
        x1_scaled = max(0, min(origin_w, x1_scaled))
        y1_scaled = max(0, min(origin_h, y1_scaled))
        x2_scaled = max(0, min(origin_w, x2_scaled))
        y2_scaled = max(0, min(origin_h, y2_scaled))

        # 7. 获取类别名称
        label_name = label_names[cls_id] if cls_id < len(label_names) else f"ID_{cls_id}"

        # 8. 封装进你原有的 BoundingBox 对象
        detected_objects.append(
            BoundingBox(cls_id, score, x1_scaled, x2_scaled, y1_scaled, y2_scaled, origin_w, origin_h, label_name)
        )

    return detected_objects