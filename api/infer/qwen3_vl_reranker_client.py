import cv2
import numpy as np
import requests
import os
import asyncio
import aiohttp
from typing import List


class Qwen3VLRerankerClient:
    def __init__(
        self,
        timeout: int = 30,
        default_instruction: str = "Retrieve text relevant to the image.",
    ):
        self.url = os.getenv("RERANKER_URL", "http://localhost:9036/v1/rerank/image-to-texts")
        self.timeout = timeout
        self.default_instruction = default_instruction
        self.session = requests.Session()  # 保留同步接口兼容性

    def rerank(self, cut_img, texts, instruction=None) -> np.ndarray:
        """同步接口（保持兼容性）"""
        if len(texts) != 2:
            raise ValueError("texts must contain exactly 2 items")

        ok, buf = cv2.imencode(".jpg", cut_img)
        if not ok:
            raise ValueError("encode image failed")

        files = {
            "file": ("crop.jpg", buf.tobytes(), "image/jpeg")
        }
        data = [
            ("texts", texts[0]),
            ("texts", texts[1]),
            ("instruction", instruction or self.default_instruction),
        ]

        resp = self.session.post(
            self.url,
            files=files,
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        result = resp.json()
        items = result.get("results", [])

        scores = np.zeros(2, dtype=np.float32)
        text_to_idx = {texts[0]: 0, texts[1]: 1}

        for item in items:
            text = item["text"]
            score = float(item["score"])
            if text in text_to_idx:
                scores[text_to_idx[text]] = score

        return scores

    async def rerank_async(self, cut_img, texts, instruction=None) -> np.ndarray:
        """异步接口（用于批量并发调用）"""
        if len(texts) != 2:
            raise ValueError("texts must contain exactly 2 items")

        ok, buf = cv2.imencode(".jpg", cut_img)
        if not ok:
            raise ValueError("encode image failed")

        form = aiohttp.FormData()
        form.add_field("file", buf.tobytes(), filename="crop.jpg", content_type="image/jpeg")
        form.add_field("texts", texts[0])
        form.add_field("texts", texts[1])
        form.add_field("instruction", instruction or self.default_instruction)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, data=form) as resp:
                resp.raise_for_status()
                result = await resp.json()

        items = result.get("results", [])

        scores = np.zeros(2, dtype=np.float32)
        text_to_idx = {texts[0]: 0, texts[1]: 1}

        for item in items:
            text = item["text"]
            score = float(item["score"])
            if text in text_to_idx:
                scores[text_to_idx[text]] = score

        return scores

    async def rerank_batch_async(self, cut_images: List[np.ndarray], texts: List[str]) -> List[np.ndarray]:
        """批量异步调用（最高效）"""
        tasks = [self.rerank_async(img, texts) for img in cut_images]
        return await asyncio.gather(*tasks, return_exceptions=True)