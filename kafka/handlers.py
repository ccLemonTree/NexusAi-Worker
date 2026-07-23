import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import aiohttp
import aioboto3
import cv2
import numpy as np
from botocore.config import Config as BotocoreConfig

from api.infer.Utils.class_info import CameraInfo
from api.infer.running import _run_analyse_sync, logic_executor as executor
from tools.concurrency import get_io_executor
from kafka.message import AnalyseInputMsg, AnalyseResultMsg, LabelResult
from tools.init import chat_infer, cfg, client
from tools.logger_tools import Kafka_Handler_logger as logger

# I/O 密集型任务专用线程池（EOS 下载、本地文件读取），与推理线程池隔离
io_executor = get_io_executor()

VLM_TIMEOUT    = float(os.getenv("VLM_TIMEOUT",    "10"))   # 大模型单次推理超时（秒）
MODEL_TIMEOUT  = float(os.getenv("MODEL_TIMEOUT",   "15"))   # 小模型单次推理超时（秒）
VECTOR_TIMEOUT = float(os.getenv("VECTOR_TIMEOUT",  "3"))   # 向量入库超时（秒）

# 全局 aioboto3 session 和 client，复用连接池
_eos_session: aioboto3.Session | None = None
_eos_client = None
_eos_client_lock = asyncio.Lock()


async def _get_eos_client():
    """获取全局 EOS S3 client（懒初始化，线程安全）"""
    global _eos_session, _eos_client

    async with _eos_client_lock:
        if _eos_client is None:
            # 自动添加协议前缀（防御性处理）
            endpoint = os.getenv("EOS_ENDPOINT", "")
            if endpoint and not endpoint.startswith(("http://", "https://")):
                endpoint = f"http://{endpoint}"

            _eos_session = aioboto3.Session(
                aws_access_key_id=os.getenv("EOS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("EOS_SECRET_KEY"),
            )

            # 创建持久化的 client（不使用 context manager）
            _eos_client = await _eos_session.client(
                "s3",
                endpoint_url=endpoint,
                config=BotocoreConfig(
                    connect_timeout=5,  # 连接超时 5 秒
                    read_timeout=30,    # 读取超时 30 秒
                    retries={'max_attempts': 2},  # 最多重试 2 次
                    max_pool_connections=50,  # 增加连接池大小，支持高并发
                ),
            ).__aenter__()

        return _eos_client


def _now_iso() -> str:
    """返回当前时间字符串，格式 yyyy-MM-dd HH:mm:ss（秒精度）"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_int(value, default: int = 0) -> int:
    """安全地把消息字段转成 int（Milvus INT64 字段用）。
    None / 空串 / 非数字字符串都退回 default，避免 DataNotMatchException。"""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"channel_id 无法转为 int：{value!r}，使用默认值 {default}")
        return default


# ---------------------------------------------------------------------------
# 图片加载
# ---------------------------------------------------------------------------

async def _download_eos_async(key: str) -> bytes:
    """
    通过 aioboto3 异步 S3 客户端从 EOS 下载对象，返回原始字节。
    使用原生 asyncio，避免 boto3 + ThreadPoolExecutor 的锁竞争。
    复用全局 client 连接池，避免每次创建新连接。
    环境变量：
        EOS_ACCESS_KEY  — AccessKey
        EOS_SECRET_KEY  — SecretKey
        EOS_ENDPOINT    — Endpoint URL，如 https://eos-wuxi-1.cmecloud.cn
        EOS_BUCKET      — Bucket 名称
    """
    s3 = await _get_eos_client()
    resp = await s3.get_object(Bucket=os.getenv("EOS_BUCKET"), Key=key)
    # resp['Body'] 是 StreamingBody，需要 read()
    return await resp["Body"].read()


async def load_image(path: str, eos: bool) -> Optional[np.ndarray]:
    """
    eos=True  → path 是 EOS 对象 Key，通过 aioboto3 异步下载
    eos=False → path 是容器内本地路径，cv2.imread 读取
    """
    import time as _time
    if eos:
        timeout = int(os.getenv("EOS_TIMEOUT", "15"))
        t0 = _time.monotonic()
        try:
            data: bytes = await asyncio.wait_for(
                _download_eos_async(path),
                timeout=timeout,
            )
            elapsed = _time.monotonic() - t0
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode 返回 None，可能不是合法图片")
            logger.info(f"EOS 图片下载成功 耗时={elapsed:.3f}s  size={len(data)}B  key={path[:120]}")
            return img
        except asyncio.TimeoutError:
            logger.error(
                f"EOS 图片下载超时(>{timeout}s)  endpoint={os.getenv('EOS_ENDPOINT')}  "
                f"bucket={os.getenv('EOS_BUCKET')}  key={path[:120]}"
            )
            return None
        except Exception as e:
            elapsed = _time.monotonic() - t0
            logger.error(
                f"EOS 图片下载失败 耗时={elapsed:.3f}s: [{type(e).__name__}] {e}  "
                f"endpoint={os.getenv('EOS_ENDPOINT')}  bucket={os.getenv('EOS_BUCKET')}  "
                f"key={path[:120]}",
                exc_info=True,
            )
            return None
    else:
        t0 = _time.monotonic()
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(io_executor, cv2.imread, path)
        elapsed = _time.monotonic() - t0
        if img is None:
            logger.error(f"本地图片读取失败 耗时={elapsed:.3f}s: {path}")
        else:
            logger.info(f"本地图片读取成功 耗时={elapsed:.3f}s  path={path}")
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
    import time as _time
    _, buf = cv2.imencode(".jpeg", img)
    img_bytes: bytes = buf.tobytes()

    loop = asyncio.get_event_loop()
    scene_start = _now_iso()
    t0 = _time.monotonic()  # 用于精确计时

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

    # 计算总耗时并记录（使用 monotonic 精确计时）
    elapsed = _time.monotonic() - t0
    logger.info(f"VLM 推理完成 耗时={elapsed:.3f}s  questions={len(questions)}条")

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
    import time as _time
    loop = asyncio.get_event_loop()
    model_start = _now_iso()
    t0 = _time.monotonic()  # 用于精确计时

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

    # 计算总耗时并记录（使用 monotonic 精确计时）
    elapsed = _time.monotonic() - t0
    logger.info(f"小模型推理完成 耗时={elapsed:.3f}s  labels={len(labels)}条")

    return results, model_start, model_end


# ---------------------------------------------------------------------------
# 向量入库（vector=true）
# ---------------------------------------------------------------------------

def _save_obj_image(obj_img: np.ndarray, device_id: str, capture_time: int) -> str:
    """
    将整张大图保存到本地，返回完整文件路径。
    目录结构：{OBJ_SAVE_PIC_LOCPATH}/{device_id}/{year}/{month}/{day}/{uuid}.jpeg
    OBJ_SAVE_PIC_LOCPATH 未设置时返回空字符串（跳过保存）。
    权限错误时返回空字符串并记录警告（降级到 EOS key）。
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

    try:
        os.makedirs(base_dir, exist_ok=True)
    except PermissionError as e:
        logger.warning(
            f"本地保存大图失败(权限不足)，降级使用 EOS key: {base_dir} — {e}"
        )
        return ""
    except Exception as e:
        logger.warning(f"本地保存大图失败: {base_dir} — {e}")
        return ""

    save_path = os.path.join(base_dir, f"{uuid.uuid4()}.jpeg")
    try:
        # cv2 是 BGR，PIL 需要 RGB
        pil_image = _PILImage.fromarray(obj_img[:, :, ::-1])
        pil_image.save(save_path, format="JPEG")
        return save_path
    except Exception as e:
        logger.warning(f"保存图片失败: {save_path} — {e}")
        return ""

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
    # channel_id 在 Milvus schema 中是 INT64，消息里可能是 str/None，需安全转 int
    channel_id   = _to_int(msg_dict.get("channel_id"), default=0)
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

        if len(boundings) > 0:
            # 保存整张大图到本地（save_local=False 时跳过，改用 EOS key）
            if save_local:
                large_image_url = _save_obj_image(img, device_id, capture_time)
                if not large_image_url:
                    logger.error(
                        f"向量入库跳过：大图保存失败 device_id={device_id} pic_url={pic_url}"
                    )
                    return False  # 保存失败，不入库
            else:
                large_image_url = pic_url

        for info in boundings:
            _, obj_img = cut_img(img, info)
            h, w, _ = obj_img.shape
            if (h * w) < pixelmax_label.get(info.classname, 20000):
                continue
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
    from datetime import datetime
    loop = asyncio.get_event_loop()
    vector_start = _now_iso()
    try:
        result: bool = await asyncio.wait_for(
            loop.run_in_executor(executor, _call_vector_sync, img, msg.dict()),
            timeout=VECTOR_TIMEOUT,
        )
        vector_end = _now_iso()

        # 计算耗时并记录
        try:
            start_dt = datetime.strptime(vector_start, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(vector_end, "%Y-%m-%d %H:%M:%S")
            elapsed = (end_dt - start_dt).total_seconds()
            status = "成功" if result else "无目标或失败"
            logger.info(f"向量入库完成 耗时={elapsed:.3f}s  status={status}  device_id={msg.deviceId}")
        except Exception:
            pass

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
