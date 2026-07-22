import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.infer.Utils.class_info import ModelClass
from api.infer.Utils.boundingbox import BoundingBox
from tools.init import cfg
from tools.logger_tools import CangQiong_Smart_Model_logger as logger
from api.infer.running import analyseRun
from api.infer.qwen3_vl_reranker_client import Qwen3VLRerankerClient
from tools.concurrency import get_triton_executor

reranker_client = Qwen3VLRerankerClient()
triton_executor = get_triton_executor()  # 使用独立的 Triton 线程池


def is_box_large_enough(box, min_size=15):
    return box.width() > min_size and box.height() > min_size


class Model(ModelClass):
    def execute(self):
        try:
            labels = cfg.logicModelDict[self.logicModelName][1]["label"]
            rer_conf = cfg.logicModelDict[self.logicModelName]["param"][2]["rer_conf"]
            text1 = cfg.logicModelDict[self.logicModelName]['rer_label'][0]
            text2 = cfg.logicModelDict[self.logicModelName]['rer_label'][1]

            # 步骤 1：批量切图
            cut_images = []
            valid_infos = []

            for info in self.logicResult:
                cut_img = self.picture[info.y1:info.y2, info.x1:info.x2]
                cut_images.append(cut_img)
                valid_infos.append(info)

            if not valid_infos:
                return [], []

            # 步骤 2：并发推理所有切图的 zuidiaoyan
            zuidiaoyan_results = []

            def run_zuidiaoyan(info, cut_img):
                result = analyseRun(labels, [cut_img, cut_img], self.cameraInfo, box_info=info, triton_executor=triton_executor)
                logger.info(f"第二次检测结果：{result}")
                result = [box for box in result if is_box_large_enough(box)]
                return result

            with ThreadPoolExecutor(max_workers=min(len(valid_infos), 16)) as pool:
                futures = {pool.submit(run_zuidiaoyan, info, cut_img): (info, cut_img)
                           for info, cut_img in zip(valid_infos, cut_images)}

                for future in as_completed(futures):
                    info, cut_img = futures[future]
                    result = future.result()
                    if len(result) > 0:
                        zuidiaoyan_results.append((info, cut_img))

            if not zuidiaoyan_results:
                return [], []

            # 步骤 3：批量异步调用 reranker（关键优化点）
            logger.info(f"开始批量 reranker，共 {len(zuidiaoyan_results)} 个")
            start_time = time.time()

            async def batch_rerank():
                tasks = []
                for info, cut_img in zuidiaoyan_results:
                    tasks.append(reranker_client.rerank_async(cut_img, [text1, text2]))
                return await asyncio.gather(*tasks, return_exceptions=True)

            # 在当前事件循环中运行异步任务
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在事件循环中，创建新任务
                    scores_list = asyncio.run(batch_rerank())
                else:
                    scores_list = loop.run_until_complete(batch_rerank())
            except RuntimeError:
                # 没有事件循环，创建新的
                scores_list = asyncio.run(batch_rerank())

            logger.info(f"批量重排序完成，耗时：{time.time() - start_time:.3f}s")

            # 步骤 4：筛选结果
            boundingboxs = []
            for (info, cut_img), scores in zip(zuidiaoyan_results, scores_list):
                if isinstance(scores, Exception):
                    logger.error(f"reranker 异常: {scores}")
                    continue

                best_idx = int(np.argmax(scores))
                best_text = text1 if best_idx == 0 else text2
                best_score = float(scores[best_idx])

                logger.info(f"best_text: {best_text}| best_score: {best_score}")

                if best_idx == 0 and best_score > rer_conf:
                    info.classname = self.logicModelName
                    boundingboxs.append(info)

            return boundingboxs, []

        except Exception as e:
            logger.error(f"{self.logicModelName} {e}", exc_info=True)
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []
