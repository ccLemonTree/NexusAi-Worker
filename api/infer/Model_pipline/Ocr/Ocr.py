import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun

class Model(ModelClass):

    def execute(self):
        try:
            res = self.logicResult[0].parames_vector["ocr_list"]
            enter_text = ""
            leave_text = ""

            # 遍历原始OCR结果（dict列表 / BoundingBox对象列表兼容）
            for item in res:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                else:
                    # BoundingBox 对象场景
                    txt = getattr(item, "text", "")

                if "进入" in txt:
                    enter_text = txt
                if "离开" in txt:
                    leave_text = txt

            # 兼容全角：： 和半角 :
            enter_num = ""
            if enter_text:
                t = enter_text.replace("：", ":")
                enter_num = t.split(":")[-1].strip()

            leave_num = ""
            if leave_text:
                t = leave_text.replace("：", ":")
                leave_num = t.split(":")[-1].strip()

            if enter_text == "" or leave_num == "":
                return [], []

            # 覆盖写入
            self.logicResult[0].parames_vector["ocr_list"] = {
                "进入": enter_num,
                "离开": leave_num
            }
            self.logicResult[0].classname = self.logicModelName

            boundingboxs = self.logicResult
            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []


