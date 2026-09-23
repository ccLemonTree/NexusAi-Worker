# -*- coding: utf-8 -*-
"""
@File    : Sticker.py
@Time    : 2026/8/25 17:27
@Author  : 陈冲
@Description : 人贴小广告
@Version : 1.0
"""
from api.infer.Utils.class_info import ModelClass
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.Utils.boundingbox import BoundingBox

class Model(ModelClass):

    def execute(self):
        try:
            boundingboxs = []
            labels = cfg.logicModelDict[self.logicModelName][0]['label']
            second_result = [box for box in self.logicResult if box.classname == labels[-1]]

            for info in self.logicResult:
                if info.classname == labels[-1]:
                    continue
                # 判断：当前目标框是否和任意人物框相邻
                matched_box = None
                for sec_box in second_result:
                    if self.is_box_adjacent(info, sec_box):
                        matched_box = sec_box
                        break
                # 只有相邻，才加入结果
                if matched_box:
                    boundingboxs.append(BoundingBox(
                        info.classID, info.confidence,
                        min(info.x1, matched_box.x1), max(info.x2, matched_box.x2),
                        min(info.y1, matched_box.y1), max(info.y2, matched_box.y2),
                        info.image_width, info.image_height, self.logicModelName,
                        parames_vector=info.parames_vector, threeDtimes=info.threeDtimes, mask=info.mask))

            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []

    @staticmethod
    def is_box_adjacent(box1, box2, dist_thresh=20):
        """
        判断两个检测框是否相邻
        box1,box2: 拥有 x1,y1,x2,y2 属性对象
        dist_thresh：像素距离阈值，框之间距离小于该值认为相邻；也可以改为IOU判断
        返回 True=相邻，False=不相邻
        """
        # 计算框1、框2的坐标
        x1_1, y1_1, x2_1, y2_1 = box1.x1, box1.y1, box1.x2, box1.y2
        x1_2, y1_2, x2_2, y2_2 = box2.x1, box2.y1, box2.x2, box2.y2

        # 计算两个框是否重叠（IOU>0）
        overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        if overlap_x > 0 and overlap_y > 0:
            return True

        # 无重叠，计算框之间最小像素距离
        dx = max(x1_1 - x2_2, x1_2 - x2_1, 0)
        dy = max(y1_1 - y2_2, y1_2 - y2_1, 0)
        min_dist = (dx ** 2 + dy ** 2) ** 0.5
        return min_dist < dist_thresh
