import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun
from api.infer.qwen3_vl_reranker_client import Qwen3VLRerankerClient

reranker_client = Qwen3VLRerankerClient()


def is_box_large_enough(box, min_size=15):
    return box.width() > min_size and box.height() > min_size


class Model(ModelClass):
    def execute(self):
        try:
            labels = cfg.logicModelDict[self.logicModelName][1]["label"]
            rer_conf = cfg.logicModelDict[self.logicModelName]["param"][2]["rer_conf"]
            text1 = cfg.logicModelDict[self.logicModelName]['rer_label'][0]
            text2 = cfg.logicModelDict[self.logicModelName]['rer_label'][1]

            logger.info(f"第一次检测结果：{labels}")

            def process_one(info):
                """处理单个检测框：yolov11推理 + rerank"""
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                result = analyseRun(labels, [cut_img, cut_img], self.cameraInfo, box_info=info)
                logger.info(f"第二次检测结果：{result}")
                result = [box for box in result if is_box_large_enough(box)]

                if len(result) == 0:
                    return None

                logger.info(f"rer_conf: {rer_conf}, text1: {text1}, text2: {text2}")

                start_time = time.time()
                scores = reranker_client.rerank(cut_img, [text1, text2])
                logger.info(f"重排序耗时：{time.time() - start_time}")

                best_idx = int(np.argmax(scores))
                best_text = text1 if best_idx == 0 else text2
                best_score = float(scores[best_idx])

                if best_idx != 0 or best_score <= rer_conf:
                    return None

                logger.info(f"best_text: {best_text}| best_score: {best_score}")
                info.classname = self.logicModelName
                return info

            # 并发处理所有检测框（原串行改为并发）
            boundingboxs = []
            if not self.logicResult:
                return boundingboxs, []
            with ThreadPoolExecutor(max_workers=min(len(self.logicResult), 10)) as pool:
                futures = {pool.submit(process_one, info): info
                           for info in self.logicResult}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        boundingboxs.append(result)

            return boundingboxs, []

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
