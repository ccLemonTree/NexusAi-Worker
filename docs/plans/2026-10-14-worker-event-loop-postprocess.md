# Worker 事件循环与 persondog 后处理优化计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 减少图片解码对事件循环的阻塞，减少 persondog NMS 的无效候选框，保持现有输出和阈值语义。

**Architecture:** 在独立分支只改 `inference/handler.py` 与 `api/infer/Triton_model/yolov5/yolov5_detector.py`。复用已有 I/O 线程池；仅按所有请求标签中的最低置信度提前过滤，保留原有类别无关 NMS、IoU 和后续逐标签阈值过滤。删除一次重复输出反序列化，不修改模型或并发配置。

**Tech Stack:** Python、asyncio、现有 ThreadPoolExecutor、NumPy、OpenCV、unittest。

---

1. 先补测试：解码在线程中执行且协程可继续运行；低阈值、阈值边界、跨类别抑制、随机样本与旧输出一致；每个输出仅反序列化一次。先验证测试失败。
2. 实现最小改动，运行隔离测试、原有服务和项目边界回归、语法检查。
3. 使用生产原镜像中的相同 NumPy/OpenCV，离线对同一组模型输出比较新旧结果及 NMS 耗时。不向 Kafka 写消息，不做向量入库，不替换在线容器。
4. 交付代码、测试与性能证据；未经进一步确认不做线上灰度、滚动发布、提交或推送。
