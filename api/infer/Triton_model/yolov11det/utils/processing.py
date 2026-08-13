from api.infer.Utils.boundingbox import BoundingBox
import cv2
import numpy as np
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


def preprocess(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img



def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, label_names=None):
    if label_names is None:
        label_names = []

    # 1. 调整输出维度 (1, 84, 8400) -> (8400, 84)
    if len(output.shape) == 3:
        pred = output[0].transpose(1, 0)
    else:
        pred = output

    # 2. 执行 NMS
    boxes = non_max_suppression(pred, conf_thres=conf_th, iou_thres=nms_threshold)

    detected_objects = []
    if len(boxes) == 0:
        return detected_objects

    # ================= 修改核心区 =================
    # 3. 计算 Letterbox 的逆映射参数
    # gain = 旧图 / 新图 的缩放比例 (取宽高中较小的那个，保持等比例)
    gain = min(input_shape[1] / origin_w, input_shape[0] / origin_h) 
    
    # pad_x, pad_y 是在 640x640 画布上单侧填充的灰边像素数
    pad_x = (input_shape[1] - origin_w * gain) / 2
    pad_y = (input_shape[0] - origin_h * gain) / 2

    for b in boxes:
        x1, y1, x2, y2, score, cls_id = b
        cls_id = int(cls_id)

        # 4. 减去灰边，并除以缩放系数，还原到原图物理尺寸
        x1_scaled = (x1 - pad_x) / gain
        y1_scaled = (y1 - pad_y) / gain
        x2_scaled = (x2 - pad_x) / gain
        y2_scaled = (y2 - pad_y) / gain

        # 5. 裁剪越界坐标（防止预测框超出原图边界）
        x1_scaled = max(0, min(origin_w, x1_scaled))
        y1_scaled = max(0, min(origin_h, y1_scaled))
        x2_scaled = max(0, min(origin_w, x2_scaled))
        y2_scaled = max(0, min(origin_h, y2_scaled))
    # ==============================================

        label_name = label_names[cls_id] if cls_id < len(label_names) else f"ID_{cls_id}"

        detected_objects.append(
            BoundingBox(cls_id, score, x1_scaled, x2_scaled, y1_scaled, y2_scaled, origin_w, origin_h, label_name)
        )

    return detected_objects



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    适配 YOLO11 的 NMS 实现
    prediction shape: (num_boxes, 4 + num_classes)
    """
    # 1. 提取框和类别分数
    # YOLO11 格式: [x, y, w, h, class0_score, class1_score, ...]
    boxes_xywh = prediction[:, :4]
    scores = prediction[:, 4:]

    # 2. 计算每个框的最大分数和对应的类别 ID
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # 3. 第一次过滤：根据置信度阈值
    mask = confidences > conf_thres
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.array([])

    # 4. 坐标转换: [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # 5. 执行 NMS (按类别独立执行)
    final_keep = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        cls_mask = (class_ids == cls)
        cls_boxes = boxes_xyxy[cls_mask]
        cls_confs = confidences[cls_mask]

        # 排序
        order = cls_confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break

            # 计算 IoU
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-7)

            # 保留 IoU 小于阈值的框
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        # 整理结果 [x1, y1, x2, y2, conf, cls_id]
        for idx in keep:
            final_keep.append([
                cls_boxes[idx, 0], cls_boxes[idx, 1],
                cls_boxes[idx, 2], cls_boxes[idx, 3],
                cls_confs[idx], cls
            ])

    return np.array(final_keep)


def xywh2xyxy(x, origin_h, origin_w, input_w, input_h):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
