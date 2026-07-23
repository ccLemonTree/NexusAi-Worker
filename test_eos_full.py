#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试完整的 EOS 下载 + 图片解码流程，模拟项目实际代码"""
import time
import numpy as np
import cv2
from boto3.session import Session

# Client 初始化
access_key = "8DEEFL5I19XFGAO558QX"
secret_key = "qkbPJ3O8kID2ovCACgIIJzlFYdC8UMVaj42N4MOH"
url = "https://eos-ningbo-1-internal.cmecloud.cn"
session = Session(access_key, secret_key)
s3_client = session.client('s3', endpoint_url=url)

bucket = "aisf-back-lsyd"
key = "2026/07/23/13/D1778555061735/D1778555061735_20260723133404.jpeg"

test_times = 20
download_times = []
decode_times = []
total_times = []

for i in range(test_times):
    try:
        # 1. 下载
        t0 = time.perf_counter()
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        data = resp['Body'].read()
        t1 = time.perf_counter()
        download_time = t1 - t0

        # 2. 解码（模拟项目代码）
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        t2 = time.perf_counter()
        decode_time = t2 - t1
        total_time = t2 - t0

        download_times.append(download_time)
        decode_times.append(decode_time)
        total_times.append(total_time)

        print(f"第{i+1:2d}次  下载={download_time:.4f}s  解码={decode_time:.4f}s  总计={total_time:.4f}s  size={len(data)}B  shape={img.shape if img is not None else 'None'}")
    except Exception as e:
        print(f"第{i+1}次失败：{e}")

if total_times:
    print("=" * 80)
    print(f"下载平均: {sum(download_times)/len(download_times):.4f}s  最大: {max(download_times):.4f}s")
    print(f"解码平均: {sum(decode_times)/len(decode_times):.4f}s  最大: {max(decode_times):.4f}s")
    print(f"总计平均: {sum(total_times)/len(total_times):.4f}s  最大: {max(total_times):.4f}s")
