# -*- coding: utf-8 -*-
"""
@File    : Parcel.py
@Time    : 2026/8/18 09:43
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""

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
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][1]['label']
            for info in self.logicResult:
                cut_img = self.picture[info.y1+20:info.y2+20, info.x1+20:info.x2+20]
                # cv2.imwrite("123.jpg", cut_img)
                result = analyseRun(labels, [cut_img, cut_img], self.cameraInfo, box_info=info)
                if len(result) !=0:
                    info.classname = self.logicModelName
                    boundingboxs.append(info)
            return boundingboxs,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [],[]

