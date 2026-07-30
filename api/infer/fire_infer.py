import os.path
from api.infer.Triton_model.triton_client import triton_inference
import cv2
import numpy as np

def showimg_scale_tools(oriimg, bbox, scale):
    """
    CV2版本：基于BoundingBox目标框，按目标宽高比调整框 + 边界扩展，绘制红色矩形
    :param oriimg: cv2 BGR图像
    :param bbox: BoundingBox实例
    :param scale: 目标宽高比 width / height
    :return: 绘制框后的图像（注意：cv2.rectangle会原地修改图像，建议外部传copy）
    """
    # 从BoundingBox读取参数
    w = bbox.image_width
    h = bbox.image_height
    box_width = bbox.width()
    box_height = bbox.height()
    xpoints, ypoints = bbox.topLeft()   # x1,y1

    box_sacel = box_width / box_height
    add_height, add_width = 0.0, 0.0

    if box_sacel > scale:
        add_height = (box_width / scale) - box_height
    elif box_sacel < scale:
        add_width = (scale * box_height) - box_width

    xpoints = float(xpoints)
    ypoints = float(ypoints)
    box_width = float(box_width)
    box_height = float(box_height)

    # 宽度方向居中扩展校正
    if (xpoints - (add_width / 2)) < 0:
        xpoints = 0.0
    elif (xpoints + box_width + (add_width / 2)) > w:
        wabscha = abs(xpoints + box_width + (add_width / 2) - w)
        xpoints = xpoints - wabscha
    else:
        xpoints = xpoints - add_width / 2

    # 高度方向居中扩展校正
    if (ypoints - (add_height / 2)) < 0:
        ypoints = 0.0
    elif (ypoints + box_height + (add_height / 2)) > h:
        wabscha = abs(ypoints + box_height + (add_height / 2) - h)
        ypoints = ypoints - wabscha
    else:
        ypoints = ypoints - add_height / 2

    box_width += add_width
    box_height += add_height

    newx1 = xpoints
    newy1 = ypoints
    newx2 = xpoints + box_width
    newy2 = ypoints + box_height

    newx1 = np.clip(newx1, 0, w - 1)
    newy1 = np.clip(newy1, 0, h - 1)
    newx2 = np.clip(newx2, 0, w - 1)
    newy2 = np.clip(newy2, 0, h - 1)

    expand = h / 18
    newx1 = int(newx1 - expand)
    newy1 = int(newy1 - expand)
    newx2 = int(newx2 + expand)
    newy2 = int(newy2 + expand)

    # cv2绘制矩形 BGR红色，线宽3
    cv2.rectangle(oriimg, (newx1, newy1), (newx2, newy2), (0, 0, 255), thickness=3)
    return oriimg

class fire_infer(object):
    def __init__(self, base_url: str, model_name: str, request_id: str = None, message: list = [],
                 mode: str = 'infer'):
        self.message = message
        self.request_id = request_id
        self.model_name = model_name
        self.mode = mode
        self.base_url = base_url + "/chat/completions"
        # 普通检测模型使用 TRITON_SERVER
        self.tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[os.getenv("TRITON_SERVER")])
        # VLM 大模型优先使用 TRITON_SERVER_VLM，未配置时回退到 TRITON_SERVER
        vlm_server = os.getenv("TRITON_SERVER_VLM") or os.getenv("TRITON_SERVER")
        self.tritonServerVLM = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[vlm_server])
    def infer(self, prompt: str, question: str, file=None):

        # 调用推理服务
        try:
            if isinstance(file, str):
                img = cv2.imread(file)
                _ = img.shape
            elif isinstance(file, bytes):
                arr = np.frombuffer(file, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = file
                _ = img.shape

        except Exception as e:
            print(f"Can not read this image ! {e}")
            return []
        result = self.tritonServer.run("yolov26det_fire", img,
                                       label_to_detect={'26_small_smoke': {'iou': 0.2, 'conf': 0.2},
                                                        '26_small_fire': {'iou': 0.2, 'conf': 0.2}})
        if len(result) != 0:
            for bounding in result:
                img = showimg_scale_tools(img,bounding,1.2)
            result_fire = self.tritonServerVLM.run("cangqiong_0.8b", img,label_to_detect={'desc_fire': {'iou': 1, 'conf': 1}})
            if not result_fire:
                return []
            result_to_return = result_fire[0]
            analyse_desc = result_to_return.parames_vector['analyse_desc']

            # 分类关键词定义，严格控制匹配优先级
            category_rules = [
                # (返回名称, 关键词列表) 从上到下优先级依次降低
                ("火情检测-大模型",
                 ["火苗", "火焰", "冒火", "燃烧", "火炬", "火把", "点燃", "明火", "火源", "火堆", "烤火", "起火"]),
                # ("动火作业检测-大模型", ["电焊", "焊接", "切割", "火花", "气割", "焊渣"]),
                ("烟雾检测-大模型", ["白烟", "白雾", "蒸汽", "雾气", "冒烟", "浓烟", "烟雾", "尾气", "排烟","粉尘"]),
            ]
            # 循环匹配，命中立即返回
            for label, keywords in category_rules:
                for kw in keywords:
                    if kw in analyse_desc:
                        for box in result:
                            box.classname = label
                            box.parames_vector = {"desc": analyse_desc,"keyword": kw}
                        return result

            # 全部未匹配返回原始文本
            return []
        return []