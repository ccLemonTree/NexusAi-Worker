# -*- coding: utf-8 -*-
"""
@File    : Charging.py
@Time    : 2026/8/18 09:43
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""

from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun


class Model(ModelClass):

    @staticmethod
    def boxes_intersect(box1, box2):
        return (
            box1.x1 < box2.x2
            and box1.x2 > box2.x1
            and box1.y1 < box2.y2
            and box1.y2 > box2.y1
        )

    def execute(self):
        try:
            labels = cfg.logicModelDict[
                self.logicModelName
            ][1]["label"]

            # 单个标签只对原图推理一次
            result = analyseRun(
                labels,
                [self.picture,self.picture],
                self.cameraInfo,
            )

            boundingboxs = []

            for info in self.logicResult:
                # 当前框与第二个标签结果相交则排除
                if any(
                    self.boxes_intersect(info, result_box)
                    for result_box in result
                ):
                    continue

                info.classname = self.logicModelName
                boundingboxs.append(info)

            return boundingboxs, []

        except Exception as e:
            logger.exception(f"{self.logicModelName} {e}")
            return [], []