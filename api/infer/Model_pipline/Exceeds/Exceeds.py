import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

class Model(ModelClass):
    def execute(self):
        try:
            height,width,channle = self.picture.shape
            k = 0
            appends = []
            for info in self.logicResult:
                info.classname = self.logicModelName
                appends.append(info)
                k+=1
            if k >=self.labelRules[self.logicModelName]["num"]:
                points = []
                for j in appends:
                    points.append([[j.x1,j.y1]])
                    points.append([[j.x2, j.y1]])
                    points.append([[j.x1, j.y2]])
                    points.append([[j.x2, j.y2]])
                x,y,w,h = cv2.boundingRect(np.array(points))
                return [BoundingBox(0, float(1.00), x,x+w,y,y+h,width,height ,self.logicModelName)],[]

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [],[]
