# NexusAi Worker

无状态图片推理 Worker。它只接收 HTTP 推理请求并返回结果，**不消费或生产 Kafka 消息**。

Kafka 的消费、调度、结果发布和 offset 提交由 [NexusAi-Dispatcher](https://github.com/ccLemonTree/nexusai-dispatcher) 负责。旧版直连 HTTP 服务仍保留在 `NexusAi` 项目中。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 存活检查 |
| `GET` | `/ready` | 就绪检查；摘流时返回 `503` |
| `POST` | `/process` | 执行一条推理任务 |

`POST /process` 必须包含：

- `Authorization: Bearer <WORKER_AUTH_TOKEN>`
- `X-Task-ID: <topic>:<partition>:<offset>`
- JSON 请求体

Worker 达到 `WORKER_MAX_CONCURRENT` 时返回 `429`；优雅退出时先将 `/ready` 变为 `503`，等待在途请求完成后退出。每个响应带有 `X-Worker-ID`、`X-Worker-Weight` 和 `X-Worker-Max-Concurrent`，供 Dispatcher 做最少连接调度。

## 本地运行

```bash
cp .env.example .env
# 按实际环境填写 .env，勿提交其中的密钥。
python -m pip install -r requirements.txt
python worker_main.py
```

服务默认监听 `0.0.0.0:8080`。运行前需保证 Triton、Milvus、VLM、EOS 等依赖可达。

## Docker Compose

一张 GPU 对应一个 Triton 和一个 Worker；默认 Compose 提供四组服务。

```bash
cp .env.example .env
docker compose --env-file .env -f docker-compose.yml -f docker-compose.multihost.yml config
docker compose --env-file .env -f docker-compose.yml -f docker-compose.multihost.yml up -d
```

多机部署时，每台服务器运行同一份 Compose，并让 Dispatcher 的 `WORKER_ENDPOINTS` 指向各服务器公开的 `8081` 至 `8084` 端口。

只构建镜像：

```bash
docker build -t nexusai-inference:local .
```

## 关键配置

| 变量 | 说明 |
| --- | --- |
| `WORKER_AUTH_TOKEN` | Dispatcher 与 Worker 共享的请求密钥，必填 |
| `WORKER_MAX_CONCURRENT` | 单个 Worker 最大在途推理数 |
| `WORKER_CAPACITY_WEIGHT_N` | 第 N 张 GPU 对应 Worker 的相对吞吐权重 |
| `TRITON_SERVER` | 当前 Worker 对应 Triton 的 gRPC 地址 |
| `MILVUS_CLIENT` | Milvus HTTP 地址 |
| `VLLM_BASE_URL` / `RERANKER_URL` | 大模型与重排序服务地址 |
| `EOS_ACCESS_KEY` / `EOS_SECRET_KEY` | 对象存储访问凭据 |

完整变量模板见 [`.env.example`](.env.example)。

## 校验

```bash
python -m unittest tests.test_inference_server tests.test_worker_project_boundary
python -m compileall inference api tools utils worker_main.py
```
