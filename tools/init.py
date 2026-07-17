import os
import importlib

from pymilvus import MilvusClient, Function, FunctionType
from api.infer.Triton_model.triton_client import triton_inference
from api.infer.Utils.config import Cfg

# 按 INFER_BACKEND 懒加载，避免其他后端的依赖（openai 包等）影响启动
_BACKEND_MAP = {
    "openai": ("api.infer.openai_infer", "openai_infer"),
    "mindie": ("api.infer.mindie_infer", "mindie_infer"),
    "fire":   ("api.infer.fire_infer",   "fire_infer"),
}

def _load_backend(name: str):
    module_path, cls_name = _BACKEND_MAP[name]
    return getattr(importlib.import_module(module_path), cls_name)

_backend_name = os.getenv("INFER_BACKEND", "fire")
chat_infer = _load_backend(_backend_name)(
    os.getenv("STRUCT_EMBEDDING_MODEL"),
    os.getenv("STRUCT_EMBEDDING_MODEL_NAME"),
)

client = MilvusClient(uri=os.getenv("MILVUS_CLIENT"), db_name=os.getenv("MILVUS_DB_NAME"))
tritonServer = triton_inference(
    os.path.join(os.getenv("NEXUSAI_HOME"), "api", "infer", "Triton_model", "weights"),
    urls=[os.getenv("TRITON_SERVER")],
)
cfg = Cfg()
