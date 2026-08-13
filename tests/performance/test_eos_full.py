#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试完整的 EOS 下载 + 图片解码流程。

需要 EOS_ACCESS_KEY、EOS_SECRET_KEY、EOS_ENDPOINT、EOS_BUCKET 和
EOS_TEST_KEY 环境变量。
"""
import os
import time

import cv2
import numpy as np
from boto3.session import Session


def main():
    session = Session(os.environ["EOS_ACCESS_KEY"], os.environ["EOS_SECRET_KEY"])
    s3_client = session.client("s3", endpoint_url=os.environ["EOS_ENDPOINT"])
    bucket = os.environ["EOS_BUCKET"]
    key = os.environ["EOS_TEST_KEY"]

    download_times = []
    decode_times = []
    total_times = []

    for i in range(20):
        try:
            t0 = time.perf_counter()
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read()
            t1 = time.perf_counter()

            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            t2 = time.perf_counter()

            download_times.append(t1 - t0)
            decode_times.append(t2 - t1)
            total_times.append(t2 - t0)

            print(
                f"第{i + 1:2d}次  下载={download_times[-1]:.4f}s  "
                f"解码={decode_times[-1]:.4f}s  总计={total_times[-1]:.4f}s  "
                f"size={len(data)}B  shape={img.shape if img is not None else 'None'}"
            )
        except Exception as exc:
            print(f"第{i + 1}次失败：{exc}")

    if total_times:
        print("=" * 80)
        print(f"下载平均: {sum(download_times) / len(download_times):.4f}s  最大: {max(download_times):.4f}s")
        print(f"解码平均: {sum(decode_times) / len(decode_times):.4f}s  最大: {max(decode_times):.4f}s")
        print(f"总计平均: {sum(total_times) / len(total_times):.4f}s  最大: {max(total_times):.4f}s")


if __name__ == "__main__":
    main()
