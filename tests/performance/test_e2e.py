"""
端到端测试：发消息 → consumer 推理 → 接收结果
用法: python test_e2e.py
"""
import asyncio
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

INPUT_TOPIC  = os.getenv("KAFKA_INPUT_TOPIC",  "model_analyse")
RESULT_TOPIC = os.getenv("KAFKA_RESULT_TOPIC", "model_analyse_result")
BOOTSTRAP    = os.getenv("KAFKA_BOOTSTRAP",    "192.168.1.115:9092")
TIMEOUT_SEC  = 120   # 等待结果最长秒数

TEST_MSG = {
    "id": int(time.time() * 1000),
    "deviceId": "TEST001",
    "presetId": 0,
    "eos": False,
    "path": r"C:\Users\chen0\Pictures\3.jpg",
    "vector": False,
    "questions": [],
    "labels": [
        {
            "alarmTypeId": 3000002,
            "inferLabels": "personnew",
            "labelDetails": '[{"conf":"0.2","desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]',
        },
        {
            "alarmTypeId": 3000001,
            "inferLabels": "fire",
            "labelDetails": '[{"conf":"0.2","desc":"置信度"},{"iou":0.2,"desc":"重叠度"}]',
        },
    ],
}


async def main():
    msg_id = TEST_MSG["id"]
    print(f"\n{'='*60}")
    print(f"  测试消息 id = {msg_id}")
    print(f"  图片路径    = {TEST_MSG['path']}")
    print(f"  labels      = personnew(3000002) + fire(3000001)")
    print(f"{'='*60}\n")

    # 1. 先订阅结果 topic，再发消息，避免竞态
    result_consumer = AIOKafkaConsumer(
        RESULT_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=f"test-result-consumer-{msg_id}",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    await result_consumer.start()
    print(f"[*] 已订阅结果 topic: {RESULT_TOPIC}")

    # 2. 发测试消息
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    await producer.start()
    await producer.send_and_wait(INPUT_TOPIC, TEST_MSG)
    await producer.stop()
    print(f"[✓] 消息已发送到 topic: {INPUT_TOPIC}")
    print(f"[*] 等待推理结果（最多 {TIMEOUT_SEC}s）...\n")

    # 3. 等待匹配 id 的结果
    deadline = time.time() + TIMEOUT_SEC
    try:
        async for record in result_consumer:
            result = record.value
            if result.get("id") == msg_id:
                print("=" * 60)
                print("  推理结果")
                print("=" * 60)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            if time.time() > deadline:
                print("[✗] 超时未收到结果，请检查 consumer 日志")
                return
    finally:
        await result_consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
