from api.infer.Utils.boundingbox import BoundingBox
import cv2
import numpy as np

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


def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5,label_names=[], letter_box=False):
    """Postprocess TensorRT outputs.
    # Args
        output: list of detections with schema
        [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()
    # Returns
        list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
    """

    # Get the num of boxes detected
    # Here we use the first row of output in that batch_size = 1
    nc = output.shape[2] - 5
    pred = np.reshape(output[0:], (-1, nc+5))
    # print(pred.shape)
    # print(output)
    # Do nms
    # print(origin_h, origin_w, input_shape[0], input_shape[1])
    boxes = non_max_suppression(pred, origin_h, origin_w, input_shape[0], input_shape[1], conf_thres=conf_th,nms_thres=nms_threshold)
    # for i in boxes:
        # print(i)
    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
    result_scores = boxes[:, 4] if len(boxes) else np.array([])
    result_classid = boxes[:, 5].astype(np.int_) if len(boxes) else np.array([])

    # indices = cv2.dnn.NMSBoxes(result_boxes, result_scores, 0.5, 0.1)
    # print(indices.shape)
    detected_objects = []
    for box, score, label in zip(result_boxes, result_scores, result_classid):
        # print(label_names[label])
        try:
            detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], origin_w, origin_h,label_names[label]))
        except Exception as e:
            print(e)
    return detected_objects

def non_max_suppression(prediction, origin_h, origin_w, input_w, input_h, conf_thres=0.5, nms_thres=0.4):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH
    boxes1 = prediction[prediction[:, 4] >= conf_thres]
    if len(boxes1):
        # 向量化操作：一次性计算所有框的类别 ID（模仿 yolov11，避免循环调用 argmax）
        class_ids = np.argmax(boxes1[:, 5:], axis=1, keepdims=True)
        boxes = np.concatenate([boxes1[:, :5], class_ids], axis=1)
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = xywh2xyxy(boxes[:, :4], origin_h, origin_w, input_w, input_h)
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)

        # 使用 OpenCV 的 NMSBoxes（C++ 实现，不受 GIL 限制）
        # 转换为 OpenCV 格式：[x, y, w, h]（向量化操作，避免循环）
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        bboxes = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()
        scores = boxes[:, 4].tolist()
        class_ids_list = boxes[:, 5].astype(int).tolist()

        # OpenCV NMSBoxes
        indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_thres, nms_thres)

        # 重建结果
        keep_boxes = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                keep_boxes.append(boxes[idx])

        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    else:
        boxes = []

    return boxes


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


