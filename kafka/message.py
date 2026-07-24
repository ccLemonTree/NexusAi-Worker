from __future__ import annotations
from typing import Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, field_validator


class QuestionEntry(BaseModel):
    system: str
    question: str


class LabelEntry(BaseModel):
    alarmTypeId: Any
    inferLabels: str          # 单标签或逗号分隔多标签，如 "personnew" 或 "fire,smoke"
    labelDetails: str         # JSON字符串：[{"conf":"0.2","desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]


class AnalyseInputMsg(BaseModel):
    id: int
    deviceId: str = ""
    presetId: int = 0
    eos: bool = False
    path: str = ""            # eos=false 时为本地路径，eos=true 时为 EOS 对象 Key
    vector: bool = False
    saveLocal: bool = True    # False = 不保存裁剪目标图到本地，仅用 EOS key 作为 large_image_url
    questions: Optional[List[QuestionEntry]] = None
    labels: Optional[List[LabelEntry]] = None
    # vector=true 时传入，用于 milvus_insert
    deviceName: str = ""
    channelId: str = ""
    channelName: str = ""
    channelNumber: str = ""
    captureTime: Optional[int] = None   # Unix 时间戳（秒），也接受 'YYYY-MM-DD HH:MM:SS' 字符串

    @field_validator("captureTime", mode="before")
    @classmethod
    def parse_capture_time(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                pass
            # 尝试解析 'YYYY-MM-DD HH:MM:SS' 格式
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return int(datetime.strptime(v, fmt).timestamp())
                except ValueError:
                    continue
        raise ValueError(f"captureTime 无法解析：{v!r}，支持格式：int 时间戳 或 'YYYY-MM-DD HH:MM:SS'")


class LabelResult(BaseModel):
    alarmTypeId: Any
    bbox: List[dict] = []


class AnalyseResultMsg(BaseModel):
    id: int
    deviceId: str = ""
    presetId: int = 0
    eos: bool = False
    path: str = ""            # eos=false 时为本地路径，eos=true 时为 EOS 对象 Key
    vector: bool = False
    saveLocal: bool = True    # False = 不保存裁剪目标图到本地，仅用 EOS key 作为 large_image_url
    questions: Optional[List[QuestionEntry]] = None
    labels: Optional[List[LabelEntry]] = None
    deviceName: str = ""
    channelId: str = ""
    channelName: str = ""
    channelNumber: str = ""
    captureTime: Optional[int] = None   # Unix 时间戳（秒）
    questionsRes: List[str] = []
    vectorRes: bool = False
    sceneStartTime: str = ""  # VLM 开始时间
    sceneTime: str = ""       # VLM 结束时间
    modelStartTime: str = ""  # 小模型开始时间
    modelTime: str = ""       # 小模型结束时间
    labelsRes: List[LabelResult] = []
