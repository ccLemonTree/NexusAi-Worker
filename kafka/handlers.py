import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import aiohttp
import cv2
import numpy as np

from api.infer.Utils.class_info import CameraInfo
from api.infer.running import _run_analyse_sync, logic_executor as executor
from kafka.message import AnalyseInputMsg, AnalyseResultMsg, LabelResult
from tools.init import chat_infer, cfg, client
from tools.logger_tools import Kafka_Handler_logger as logger

VLM_TIMEOUT    = float(os.getenv("VLM_TIMEOUT",    "60"))   # 大模型单次推理超时（秒）
MODEL_TIMEOUT  = float(os.getenv("MODEL_TIMEOUT",   "30"))   # 小模型单次推理超时（秒）
VECTOR_TIMEOUT = float(os.getenv("VECTOR_TIMEOUT",  "30"))   # 向量入库超时（秒）


def _now_iso() -> str:
    """返回 UTC 时间 ISO8601 字符串，精确到毫秒。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# 图片加载
# ---------------------------------------------------------------------------

def _download_eos_sync(key: str) -> bytes:
    """
    通过 boto3 S3 客户端从 EOS 下载对象，返回原始字节。
    环境变量：
        EOS_ACCESS_KEY  — AccessKey
        EOS_SECRET_KEY  — SecretKey
        EOS_ENDPOINT    — Endpoint URL，如 https://eos-wuxi-1.cmecloud.cn
        EOS_BUCKET      — Bucket 名称
    """
    import boto3
    from boto3.session import Session as _S3Session

    session = _S3Session(
        os.getenv("EOS_ACCESS_KEY"),
        os.getenv("EOS_SECRET_KEY"),
    )
    s3 = session.client("s3", endpoint_url=os.getenv("EOS_ENDPOINT"))
    resp = s3.get_object(Bucket=os.getenv("EOS_BUCKET"), Key=key)
    return resp["Body"].read()


async def load_image(path: str, eos: bool) -> Optional[np.ndarray]:
    """
    eos=True  → path 是 EOS 对象 Key，通过 boto3 S3 客户端下载
    eos=False → path 是容器内本地路径，cv2.imread 读取
    """
    if eos:
        try:
            timeout = int(os.getenv("EOS_TIMEOUT", "15"))
            loop = asyncio.get_event_loop()
            data: bytes = await asyncio.wait_for(
                loop.run_in_executor(executor, _download_eos_sync, path),
                timeout=timeout,
            )
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode 返回 None，可能不是合法图片")
            return img
        except Exception as e:
            logger.error(f"EOS 图片下载失败: {e}  key={path[:120]}")
            return None
    else:
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(executor, cv2.imread, path)
        if img is None:
            logger.error(f"本地图片读取失败: {path}")
        return img


# ---------------------------------------------------------------------------
# VLM 推理
# ---------------------------------------------------------------------------

def _call_vlm_sync(system: str, question: str, img_bytes: bytes) -> str:
    """在线程池中同步调用 chat_infer，避免阻塞事件循环。"""
    try:
        return chat_infer.infer(system, question, file=img_bytes)
    except Exception as e:
        logger.error(f"VLM 推理失败: {e}")
        return "推理异常"


async def run_vlm_tasks(
    img: np.ndarray, questions: list
) -> Tuple[List[str], str, str]:
    """
    并发调用 VLM，每条 question 独立推理。
    返回 (结果列表, sceneStartTime, sceneTime)
    超时时间由 VLM_TIMEOUT 环境变量控制（默认 60s）。
    """
    _, buf = cv2.imencode(".jpeg", img)
    img_bytes: bytes = buf.tobytes()

    loop = asyncio.get_event_loop()
    scene_start = _now_iso()

    tasks = [
        loop.run_in_executor(executor, _call_vlm_sync, q["system"], q["question"], img_bytes)
        for q in questions
    ]
    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=VLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"VLM 推理超时（>{VLM_TIMEOUT}s），共 {len(questions)} 条 question")
        return ["推理超时"] * len(questions), scene_start, _now_iso()

    scene_end = _now_iso()
    results: List[str] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            logger.error(f"VLM task[{i}] 异常: {r}")
            results.append("推理异常")
        else:
            results.append(r)
    return results, scene_start, scene_end


# ---------------------------------------------------------------------------
# 小模型推理
# ---------------------------------------------------------------------------

def _call_model_sync(img: np.ndarray, entry: dict) -> dict:
    """
    在线程池中同步执行单条 label 的小模型推理。
    entry 对应 LabelEntry.dict()
    """
    alarm_type_id = entry["alarmTypeId"]
    try:
        # inferLabels 可能是逗号分隔的多标签
        infer_labels = {
            lbl.strip() for lbl in entry["inferLabels"].split(",") if lbl.strip()
        }
        # labelDetails 格式：[{"conf":"0.2","desc":"..."}, {"iou":0.2,"desc":"..."}]
        details = json.loads(entry["labelDetails"])
        conf = float(details[0].get("conf", 0.5))
        iou  = float(details[1].get("iou",  0.2))
        label_rules = {lbl: {"conf": conf, "iou": iou} for lbl in infer_labels}

        camera_info = CameraInfo()
        camera_info.imgsList = [img]
        camera_info.deviceId  = ""
        camera_info.presetId  = "0"

        result = _run_analyse_sync(camera_info, infer_labels, label_rules)

        # 展平所有 bbox，统一挂上 alarmTypeId
        all_bbox = []
        for v in result.values():
            all_bbox.extend(v.get("bbox", []))

        return {"alarmTypeId": alarm_type_id, "bbox": all_bbox}

    except Exception as e:
        logger.error(f"小模型推理失败 alarmTypeId={alarm_type_id}: {e}", exc_info=True)
        return {"alarmTypeId": alarm_type_id, "bbox": []}


async def run_model_tasks(
    img: np.ndarray, labels: list
) -> Tuple[List[dict], str, str]:
    """
    并发执行多条 label 的小模型推理，每条带自己的 alarmTypeId。
    返回 (结果列表, modelStartTime, modelTime)
    超时时间由 MODEL_TIMEOUT 环境变量控制（默认 30s）。
    """
    loop = asyncio.get_event_loop()
    model_start = _now_iso()

    tasks = [
        loop.run_in_executor(executor, _call_model_sync, img, entry)
        for entry in labels
    ]
    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=MODEL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"小模型推理超时（>{MODEL_TIMEOUT}s），共 {len(labels)} 条 label")
        return (
            [{"alarmTypeId": e["alarmTypeId"], "bbox": []} for e in labels],
            model_start,
            _now_iso(),
        )

    model_end = _now_iso()
    results: List[dict] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            logger.error(f"model task[{i}] 异常: {r}")
            results.append({"alarmTypeId": labels[i]["alarmTypeId"], "bbox": []})
        else:
            results.append(r)
    return results, model_start, model_end


# ---------------------------------------------------------------------------
# 向量入库（vector=true）
# ---------------------------------------------------------------------------

def _save_obj_image(obj_img: np.ndarray, device_id: str, capture_time: int) -> str:
    """
    将裁剪目标图保存到本地，返回完整文件路径。
    目录结构：{OBJ_SAVE_PIC_LOCPATH}/{device_id}/{year}/{month}/{day}/{uuid}.jpeg
    OBJ_SAVE_PIC_LOCPATH 未设置时返回空字符串（跳过保存）。
    """
    import uuid
    from datetime import datetime
    from PIL import Image as _PILImage

    obj_root = os.getenv("OBJ_SAVE_PIC_LOCPATH", "")
    if not obj_root:
        return ""

    dt = datetime.fromtimestamp(capture_time)
    base_dir = os.path.join(
        obj_root, device_id,
        dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"),
    )
    os.makedirs(base_dir, exist_ok=True)

    save_path = os.path.join(base_dir, f"{uuid.uuid4()}.jpeg")
    # cv2 是 BGR，PIL 需要 RGB
    pil_image = _PILImage.fromarray(obj_img[:, :, ::-1])
    pil_image.save(save_path, format="JPEG")
    return save_path

def _call_vector_sync(img: np.ndarray, msg_dict: dict) -> bool:
    """
    复用 vec2milvus 的核心逻辑：检测目标 → 生成向量 → 写入 Milvus。
    此函数在线程池中同步运行（gme_vector 内部有 asyncio.run，需隔离）。
    msg_dict 是 AnalyseInputMsg.dict()，避免跨线程传递 Pydantic 对象。
    返回 True 表示至少成功入库一条，False 表示无目标或入库异常。
    """
    import asyncio as _asyncio
    from api.infer.running import analyseRun
    from api.vector.vector import gme_vector
    from tools.api import milvus_insert
    from apps.Cangqiong_Smart_Search.utils.utils import cut_img
    from datetime import datetime

    device_id    = msg_dict.get("deviceId", "")
    device_name  = msg_dict.get("device_name", "")
    channel_id   = msg_dict.get("channel_id", "")
    channel_name = msg_dict.get("channel_name", "")
    channel_num  = msg_dict.get("channel_number", "")
    pic_url      = msg_dict.get("path", "")          # EOS 对象 Key
    save_local   = msg_dict.get("save_local", True)  # False = 不保存目标图到本地

    # capture_time：优先用消息里的 int 时间戳，否则用当前时间
    raw_ts = msg_dict.get("capture_time")
    if raw_ts:
        capture_time = int(raw_ts)
        partition_name = f"p_{datetime.fromtimestamp(capture_time).strftime('%Y%m%d')}"
    else:
        now = datetime.now()
        capture_time = int(now.timestamp())
        partition_name = f"p_{now.strftime('%Y%m%d')}"

    try:
        setsLabel = cfg.nexusDict["Smart_Search"]["basic_label"]
        boundings = analyseRun(setsLabel, [img])

        pixelmax_label = {"face": 1000, "PlateSearch-car": 1000}
        inserted = False

        for info in boundings:
            _, obj_img = cut_img(img, info)
            h, w, _ = obj_img.shape
            if (h * w) < pixelmax_label.get(info.classname, 20000):
                continue

            # 保存裁剪目标图到本地（save_local=False 时跳过，改用 EOS key）
            if save_local:
                large_image_url = _save_obj_image(obj_img, device_id, capture_time)
                if not large_image_url:
                    large_image_url = pic_url   # OBJ_SAVE_PIC_LOCPATH 未配置时降级
            else:
                large_image_url = pic_url

            # gme_vector 是 async 函数，在线程中用独立事件循环调用
            vector, emb_type = _asyncio.run(
                gme_vector(
                    question="",
                    pic_path=obj_img,
                    prompt=os.getenv("SYSTEM_PROMPT", ""),
                    insert_type=info.classname,
                )
            )
            milvus_insert(
                client,
                coll_name=os.getenv("MILVUS_VECTOR_FILTER_COLLECTION_NAME"),
                partition_name=partition_name,
                desc="",
                image_url=pic_url,
                large_image_url=large_image_url,
                vector=vector,
                device_id=device_id,
                capture_time=capture_time,
                device_name=device_name,
                channel_id=channel_id,
                channel_name=channel_name,
                channel_number=channel_num,
                x1=info.u1, x2=info.u2, y1=info.v1, y2=info.v2,
                target_category=info.classname,
                search_type="obj",
            )
            inserted = True

        return inserted

    except Exception as e:
        logger.error(f"向量入库失败 device_id={device_id}: {e}", exc_info=True)
        return False


async def run_vector_task(img: np.ndarray, msg: AnalyseInputMsg) -> bool:
    loop = asyncio.get_event_loop()
    try:
        result: bool = await asyncio.wait_for(
            loop.run_in_executor(executor, _call_vector_sync, img, msg.dict()),
            timeout=VECTOR_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"向量入库超时（>{VECTOR_TIMEOUT}s）device_id={msg.deviceId}")
        return False


# ---------------------------------------------------------------------------
# 消息总处理入口
# ---------------------------------------------------------------------------

async def process_message(raw: bytes) -> Optional[dict]:
    """
    消费一条 Kafka 消息，返回结果 dict；解析/图片加载失败时返回 None（跳过该消息）。
    """
    # 1. 解析消息
    try:
        data = json.loads(raw.decode("utf-8"))
        msg = AnalyseInputMsg(**data)
    except Exception as e:
        logger.error(f"消息解析失败: {e}  raw={raw[:200]}")
        return None

    logger.info(f"Processing id={msg.id}  deviceId={msg.deviceId}  eos={msg.eos}")

    # 2. 加载图片
    img = await load_image(msg.path, msg.eos)
    if img is None:
        logger.error(f"id={msg.id} 图片加载失败，跳过")
        return None

    # 3. 并发执行三路任务
    questions_result: List[str] = []
    scene_start = scene_end = ""
    labels_result: List[dict] = []
    model_start = model_end = ""
    vector_ok: bool = False

    coros = []
    tags  = []

    if msg.questions:
        coros.append(run_vlm_tasks(img, [q.dict() for q in msg.questions]))
        tags.append("vlm")

    if msg.labels:
        coros.append(run_model_tasks(img, [l.dict() for l in msg.labels]))
        tags.append("model")

    if msg.vector:
        coros.append(run_vector_task(img, msg))
        tags.append("vector")

    gathered = await asyncio.gather(*coros, return_exceptions=True)

    for tag, result in zip(tags, gathered):
        if isinstance(result, Exception):
            logger.error(f"id={msg.id} [{tag}] 任务异常: {result}", exc_info=True)
            continue
        if tag == "vlm":
            questions_result, scene_start, scene_end = result
        elif tag == "model":
            labels_result, model_start, model_end = result
        elif tag == "vector":
            vector_ok = bool(result)  # True=至少入库一条，False=无目标或异常

    # 4. 组装结果
    output = AnalyseResultMsg(
        id=msg.id,
        questions=questions_result,
        vector=vector_ok,
        sceneStartTime=scene_start,
        sceneTime=scene_end,
        modelStartTime=model_start,
        modelTime=model_end,
        labels=[LabelResult(**r) for r in labels_result],
    )

    logger.info(f"Done id={msg.id}  questions={len(questions_result)}  labels={len(labels_result)}")
    return output.dict()
