# -*- coding: utf-8 -*-
"""
@File    : Dog.py
@Time    : 2026/9/4 16:28
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""

from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger


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
            labels = cfg.logicModelDict[self.logicModelName][0]['label']
            dogs = [box for box in self.logicResult if box.classname == labels[0]]
            people = [box for box in self.logicResult if box.classname == labels[1]]
            leashes = [box for box in self.logicResult if box.classname == labels[2]]
            boundingboxs = []
            for dog in dogs:
                is_leashed = any(
                    self.boxes_intersect(dog, leash)
                    and any(self.boxes_intersect(leash, person) for person in people)
                    for leash in leashes
                )
                if not is_leashed:
                    dog.classname = self.logicModelName
                    boundingboxs.append(dog)
            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
