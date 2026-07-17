from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel


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
    questions: Optional[List[QuestionEntry]] = None
    labels: Optional[List[LabelEntry]] = None
    # vector=true 时传入，用于 milvus_insert
    device_name: str = ""
    channel_id: str = ""
    channel_name: str = ""
    channel_number: str = ""
    capture_time: Optional[int] = None   # Unix 时间戳（秒）


class LabelResult(BaseModel):
    alarmTypeId: Any
    bbox: List[dict] = []


class AnalyseResultMsg(BaseModel):
    id: int
    questions: List[str] = []
    vector: bool = False
    sceneStartTime: str = ""  # VLM 开始时间
    sceneTime: str = ""       # VLM 结束时间
    modelStartTime: str = ""  # 小模型开始时间
    modelTime: str = ""       # 小模型结束时间
    labels: List[LabelResult] = []
