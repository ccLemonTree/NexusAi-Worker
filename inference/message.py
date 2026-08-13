from __future__ import annotations
from typing import Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, field_validator


def _to_datetime_str(v, nullable: bool = False) -> Optional[str]:
    """将任意时间格式统一转为 'YYYY-MM-DD HH:MM:SS'。

    支持：
      - int / float   Unix 时间戳（秒）
      - str 数字      纯数字字符串，当作 Unix 时间戳
      - str 日期时间  'YYYY-MM-DD HH:MM:SS' / ISO 8601 等常见格式
      - None / ''     nullable=True 返回 None，否则返回 ''
    """
    if v is None or v == "":
        return None if nullable else ""
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(int(v)).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, str):
        # 纯数字字符串 → Unix 时间戳
        if v.strip().isdigit():
            return datetime.fromtimestamp(int(v.strip())).strftime("%Y-%m-%d %H:%M:%S")
        # 逐一尝试常见格式
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y/%m/%d %H:%M:%S",
            "%Y%m%d%H%M%S",
        ):
            try:
                return datetime.strptime(v.strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    raise ValueError(f"无法解析时间：{v!r}")


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
    captureTime: Optional[str] = None  # 统一格式 'YYYY-MM-DD HH:MM:SS'
    snapshotTime: str = ""             # 统一格式 'YYYY-MM-DD HH:MM:SS'

    @field_validator("captureTime", mode="before")
    @classmethod
    def normalize_capture_time(cls, v):
        return _to_datetime_str(v, nullable=True)

    @field_validator("snapshotTime", mode="before")
    @classmethod
    def normalize_snapshot_time(cls, v):
        return _to_datetime_str(v, nullable=False)


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
    captureTime: Optional[str] = None  # 统一格式 'YYYY-MM-DD HH:MM:SS'
    snapshotTime: str = ""             # 统一格式 'YYYY-MM-DD HH:MM:SS'
    error: str = ""           # 错误信息，空字符串表示成功
    questionsRes: List[List[dict]] = []  # 每个 question 对应一组 BoundingBox.dict()
    vectorRes: bool = False
    sceneStartTime: str = ""  # VLM 开始时间
    sceneTime: str = ""       # VLM 结束时间
    modelStartTime: str = ""  # 小模型开始时间
    modelTime: str = ""       # 小模型结束时间
    labelsRes: List[LabelResult] = []
