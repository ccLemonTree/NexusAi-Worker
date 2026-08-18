# -*- coding: utf-8 -*-
"""
@File    : Color.py.py
@Time    : 2026/8/18 09:42
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""

import cv2
import numpy as np

from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun


class Model(ModelClass):

    @staticmethod
    def get_plate_color(image):
        if image is None or image.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 排除白色字符、灰色区域及过暗像素
        valid = (s >= 50) & (v >= 40)
        green_count = np.count_nonzero(valid & (h >= 35) & (h < 90))
        blue_count = np.count_nonzero(valid & (h >= 90) & (h <= 135))
        color_count = green_count + blue_count

        # 有效颜色像素太少，不进行判断
        if color_count < 100 or color_count / image.shape[0] / image.shape[1] < 0.15:
            return "unknown"

        if green_count / color_count >= 0.55:
            return "green"
        if blue_count / color_count >= 0.55:
            return "blue"

        return "unknown"

    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[
                self.logicModelName
            ][1]["label"]

            if not labels:
                logger.error(f"{self.logicModelName} label 配置为空")
                return [], []

            target_color = labels[0].strip().lower()

            for info in self.logicResult:
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                plate_color = self.get_plate_color(cut_img)

                if plate_color in target_color:
                    info.classname = self.logicModelName
                    boundingboxs.append(info)

            return boundingboxs, []

        except Exception as e:
            logger.exception(f"{self.logicModelName} {e}")
            return [], []