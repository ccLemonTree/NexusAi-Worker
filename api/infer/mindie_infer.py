# -*- coding: utf-8 -*-
# @Time    : 2025/6/5 15:54:21
# @Author  : 陈澔麟
# @File    : mindie_infer.py

import requests
from api.infer.utils import file2base64img
import json


class mindie_infer(object):
    def __init__(self, base_url: str, model_name: str, request_id: str = None, message: list = [],
                 mode: str = 'infer'):
        self.message = message
        self.request_id = request_id
        self.model_name = model_name
        self.mode = mode
        self.base_url = base_url + "/chat/completions"

    def infer(self, prompt: str, question: str, file=None):
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": file2base64img(file,prefix=False)}
                ]
            }
        ]
        if self.mode == 'infer':
            messages = messages

        client = requests.post(
            self.base_url, json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 100
            }
        )
        chat_response = client.json()
        result = chat_response['choices'][0]['message']['content']
        return result


