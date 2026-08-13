#!/usr/bin/env python3
"""
诊断脚本：检查线程池配置和推理瓶颈
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

print("="*80)
print("环境变量检查")
print("="*80)
print(f"ANALYSE_MAX_WORKERS: {os.getenv('ANALYSE_MAX_WORKERS', '未设置')}")
print(f"LOGIC_MAX_WORKERS: {os.getenv('LOGIC_MAX_WORKERS', '未设置')}")
print(f"TRITON_POOL_SIZE: {os.getenv('TRITON_POOL_SIZE', '未设置')}")
print(f"MODEL_TIMEOUT: {os.getenv('MODEL_TIMEOUT', '未设置')}")
print(f"CPU count: {os.cpu_count()}")
print()

print("="*80)
print("线程池实际配置")
print("="*80)

from tools.concurrency import get_inference_executor, get_logic_executor, get_io_executor

inference_executor = get_inference_executor()
logic_executor = get_logic_executor()
io_executor = get_io_executor()

print(f"inference_executor workers: {inference_executor._max_workers}")
print(f"logic_executor workers: {logic_executor._max_workers}")
print(f"io_executor workers: {io_executor._max_workers}")
print()

print("="*80)
print("Triton 连接池配置")
print("="*80)

from tools.init import tritonServer
try:
    pool_size = tritonServer._pool.maxsize
except AttributeError:
    pool_size = "未知"
print(f"Triton pool size: {pool_size}")
print()

print("="*80)
print("分析")
print("="*80)

inference_workers = inference_executor._max_workers
logic_workers = logic_executor._max_workers

print(f"当前配置:")
print(f"  - inference_executor: {inference_workers} 线程")
print(f"  - logic_executor: {logic_workers} 线程")
print()

if inference_workers < 16:
    print(f"⚠️  警告：inference_executor 只有 {inference_workers} 个线程")
    print(f"   建议：设置环境变量 ANALYSE_MAX_WORKERS=32 或更高")
    print()

if logic_workers < 16:
    print(f"⚠️  警告：logic_executor 只有 {logic_workers} 个线程")
    print(f"   建议：设置环境变量 LOGIC_MAX_WORKERS=32 或更高")
    print()

print("="*80)
print("推理流程分析")
print("="*80)
print("Smoking-handsmoking 推理流程:")
print()
print("1. run_model_tasks (inference/handler.py)")
print("   - 使用 inference_executor (analyseRun)")
print("   - 当前线程数:", inference_workers)
print()
print("2. analyseRun (api/infer/running.py)")
print("   - 调用 Unlogic_run")
print("   - 使用 inference_executor 提交 Triton 推理")
print()
print("3. logic_run (api/infer/Utils/result_utils.py)")
print("   - 使用 logic_executor")
print("   - 当前线程数:", logic_workers)
print()

if inference_workers < 16 or logic_workers < 16:
    print("⚠️  瓶颈分析:")
    print(f"   16 并发请求需要 {inference_workers} 个线程处理")
    print(f"   → 超出部分会排队等待")
    print(f"   → 导致延迟从 2s 增加到 11s")
    print()
    print("✅  解决方案:")
    print("   docker run 时添加:")
    print("   -e ANALYSE_MAX_WORKERS=32 \\")
    print("   -e LOGIC_MAX_WORKERS=32")
