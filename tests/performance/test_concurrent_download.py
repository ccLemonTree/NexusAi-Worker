#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟推理 worker 的并发下载场景：
16 个并发任务，每个任务在独立线程池中下载图片
"""
import asyncio
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from boto3.session import Session

# 创建线程池（模拟 io_executor）
io_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="io")


def download_sync():
    """同步下载（在线程池中执行）"""
    session = Session(os.environ["EOS_ACCESS_KEY"], os.environ["EOS_SECRET_KEY"])
    s3 = session.client('s3', endpoint_url=os.environ["EOS_ENDPOINT"])
    resp = s3.get_object(Bucket=os.environ["EOS_BUCKET"], Key=os.environ["EOS_TEST_KEY"])
    return resp['Body'].read()


async def download_one(task_id: int):
    """模拟一个推理请求的下载流程"""
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    # 模拟 load_image 的调用方式
    data = await loop.run_in_executor(io_executor, download_sync)

    elapsed = time.perf_counter() - t0
    print(f"Task {task_id:2d}  下载耗时={elapsed:.3f}s  size={len(data)}B")
    return elapsed


async def main():
    print(f"开始测试：16 个并发任务从 {io_executor._max_workers} 线程池下载")
    print("=" * 60)

    t_start = time.perf_counter()

    # 16 个并发任务（模拟 WORKER_MAX_CONCURRENT=16）
    tasks = [download_one(i) for i in range(16)]
    results = await asyncio.gather(*tasks)

    t_total = time.perf_counter() - t_start

    print("=" * 60)
    print(f"总耗时: {t_total:.3f}s")
    print(f"平均每个任务: {sum(results)/len(results):.3f}s")
    print(f"最慢任务: {max(results):.3f}s")
    print(f"最快任务: {min(results):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
