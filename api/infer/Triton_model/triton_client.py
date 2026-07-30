
import os
import json
import cv2, sys, traceback
import tritonclient.grpc as grpcclient
from api.infer.Utils.Singletons import Singleton
from api.infer.Utils.boundingbox import BoundingBox
from api.infer.Triton_model.yolov5.yolov5_detector import yolov5
from api.infer.Triton_model.retina.retina_face import retina
from api.infer.Triton_model.dlib.dlib_face import dlib_face_vector
from api.infer.Triton_model.plate.plate_det import plate_det
from api.infer.Triton_model.resnest.resnest_detection import resnest
from api.infer.Triton_model.sam3.sam3_detector import sam3
from api.infer.Triton_model.yolov11det.yolov11_detector import yolov11det
from api.infer.Triton_model.yolov11cls.yolov11cls_detector import yolov11cls
from api.infer.Triton_model.fastvlm.fastvlm_detector import fastvlm_detector
from tools.concurrency import get_triton_pool, get_triton_pool_vlm, get_triton_pool_sam3
from api.infer.Triton_model.yolov26det.yolov26_detector import yolov26det
from api.infer.Triton_model.cangqiong.cangqiong import cangqiong_detector
def none(a,b,c,d,e, box_info=""):
    return [],[]

@Singleton
class triton_inference:
    def __init__(self, config_dir, urls):
        self.config_dir = config_dir
        self.init_data = self.initialize(self.config_dir)
        self.service_scheduling = {}
        self. urls = urls
        self._pool = get_triton_pool()
        self._pool_vlm = get_triton_pool_vlm()          # 大模型(fastvlm等) 专用池
        self._pool_sam3 = get_triton_pool_sam3()        # SAM3 专用池
        self.model_class = {
            "yolov5": yolov5,
            "retina": retina,
            "dlib": dlib_face_vector,
            "plate": plate_det,
            "resnest": resnest,
            "sam3": sam3,
            "fastvlm": fastvlm_detector,
            "yolov11det": yolov11det,
            "yolov11cls":yolov11cls,
            "yolov26det":yolov26det,
            "cangqiong": cangqiong_detector,
            "none": none,
        }
    # 索检代码仓库
    def initialize(self, dir):
        dir_list = []
        files = os.listdir(dir)
        for i in files:
            if os.path.isfile(i):
                print("file")
            else:
                dir_list.append(i)
        init_data = dict()
        for i in dir_list:
            json_dir = dir + r"/" + i + r"/" + "config.json"
            with open(json_dir, 'r', encoding='utf8') as fp:
                json_data_init = {}
                json_data = json.load(fp)
                name = json_data["name"]
                iou_thres = json_data["iou_thres"]
                conf_thres = json_data["conf_thres"]
                input = json_data["input"]
                output = json_data["output"]
                classes = json_data["classes"]
                version_policy = str(json_data["model_version"][-1])
                json_data_init["name"] = name
                json_data_init["label_names"] = classes
                json_data_init["input"] = input
                json_data_init["output"] = output
                json_data_init["model_version"] = version_policy
                json_data_init["iou_thres"] = iou_thres
                json_data_init["conf_thres"] = conf_thres
                init_data[str(json_data["name"])] = json_data_init
        return init_data

    # 查看服务可检测标签
    def detected_labels(self, service_name):
        for key, value in self.init_data.items():
            # print(key,service_name)
            if key == service_name:
                label_names = value["label_names"]
        return label_names


    def run(self, service_name, images, camerInfo=None, label_to_detect={},
            box_info=BoundingBox(0, 0, 0, 0, 0, 0, 1, 1, "cls")):
        """
        param:
        service_name : 推理的模型名称
        images : 单张图片
        camerInfo : 设备信息
        label_to_detect ： 传入的分析标签
        box_info: 传入的boundingbox
        Return : [BoundingBox , ...]
        """

        result_to_return = []

        try:
            result = isinstance(images, str)
            if result:
                img = cv2.imread(images)
                img.shape

            else:
                img = images
                img.shape
        except Exception as e:
            print(f"Can not read this image !{e}")
            return []

        try:
            name = service_name.split("_")[0]

            # 根据模型类型选择专用连接池
            if name == "sam3":
                pool = self._pool_sam3
            elif name in ("cangqiong", "fastvlm"):
                pool = self._pool_vlm
            else:
                pool = self._pool

            with pool.borrow() as client:
                if client is None:
                    return []
                result_to_return, _ = self.model_class.get(name, "none")(
                    client, service_name, self.init_data, img, label_to_detect, box_info=box_info)

            return result_to_return
        except Exception as e:
            print(f"Service not enabled{service_name}")
            print(e)
            print(sys.exc_info())
            print('\n', '>>>' * 20, service_name)
            print(traceback.print_exc())
            print('\n', '>>>' * 20)
            print(traceback.format_exc())
            return []

    def fire_run(self, service_name, images, camerInfo=None, label_to_detect={},
            box_info=BoundingBox(0, 0, 0, 0, 0, 0, 1, 1, "cls")):
        result_to_return = []
        try:
            if isinstance(images, str):
                img = cv2.imread(images)
                _ = img.shape
            elif isinstance(images, bytes):
                img = images
                if len(img) == 0:
                    raise ValueError("Bytes 数据为空")
            else:
                img = images
                _ = img.shape

        except Exception as e:
            print(f"Can not read this image ! {e}")
            return []

        try:
            name = service_name.split("_")[0]
            func = self.model_class.get(name, none)
            if func == none:
                return []

            with self._pool_vlm.borrow() as client:
                if client is None:
                    return [], {}   # 连接池耗尽，返回空结果
                result_to_return, time_client_json = self.model_class.get(name, "none")(
                    client, service_name, self.init_data, img, label_to_detect, box_info=box_info)
            return result_to_return
        except Exception as e:
            print(f"Async Service not enabled {service_name}")
            print(e)
            print(sys.exc_info())
            print('\n', '>>>' * 20, service_name)
            print(traceback.print_exc())
            print('\n', '>>>' * 20)
            print(traceback.format_exc())
            return []

