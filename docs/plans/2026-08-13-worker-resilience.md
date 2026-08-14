# Worker 可靠性实施计划

> **给 Claude：** 实施时必须使用 `test-driven-development` 子技能，并逐项完成本计划。

**目标：** 在不改变现有 Kafka 职责边界和分区内顺序的前提下，增加容量感知重试、自动熔断、readiness 摘流和 Worker 优雅关闭能力。

**架构：** `NexusAi-Dispatcher` 仍是唯一的 Kafka 消费者和结果生产者。它通过 Service 发送推理请求，收到容量响应时立即使用新连接重试，基础设施故障时执行退避，并临时熔断故障目标。`NexusAi` 保持无状态，提供 liveness/readiness，并在关闭时等待在途 HTTP 请求完成。

**技术栈：** Python 3.12、asyncio、aiohttp、aiokafka、unittest、Docker Compose。

---

### 任务 1：Dispatcher 容量重试与熔断器

**涉及文件：**
- 创建：`../NexusAi-Dispatcher/tests/test_worker_client.py`
- 修改：`../NexusAi-Dispatcher/dispatcher/worker_client.py`
- 修改：`../NexusAi-Dispatcher/.env.example`
- 修改：`../NexusAi-Dispatcher/docker-compose.yml`

1. 添加异步测试，证明 `429` 或容量类 `503` 会立即重试，并且不会增加基础设施故障计数。
2. 添加异步测试，证明连接失败达到配置阈值后会熔断目标，冷却期间会跳过该目标，冷却结束后只允许一次半开探测。
3. 只运行 `python -m unittest tests.test_worker_client -v`，确认新测试因缺少对应行为而失败。
4. 使用单调时钟实现最小化的按目标熔断状态；正常成功连接保持可复用，失败或容量响应在重试前关闭。
5. 再次运行相同的重点测试并确认通过。

### 任务 2：保持 Kafka 顺序与背压

**涉及文件：**
- 修改：`../NexusAi-Dispatcher/tests/test_consumer.py`
- 仅在必要时修改：`../NexusAi-Dispatcher/dispatcher/consumer.py`

1. 添加测试验证既定顺序：先取得 Worker 结果，再收到 Kafka 结果确认，最后提交输入 offset。
2. 添加测试，验证失败时会回退当前 offset，并且绝不会错误推进分区进度。
3. 每个分区只保留一个在途任务；不增加应用队列或并发 offset 前沿跟踪器。
4. 获得用户明确授权后，只运行相关的 Consumer 测试。

### 任务 3：Worker readiness 与优雅摘流

**涉及文件：**
- 创建：`tests/test_inference_server.py`
- 修改：`inference_server.py`
- 修改：`worker_main.py`
- 修改：`docker-compose.yml`

1. 添加测试，证明接收流量时 `/ready` 返回 200，开始摘流后返回 503。
2. 添加测试，证明摘流时会拒绝新的 `/process` 请求，而已经接收的请求仍能执行完成。
3. 只运行 `python -m unittest tests.test_inference_server -v`，确认新测试因缺少对应行为而失败。
4. 在 aiohttp 应用中跟踪是否接收请求及在途请求状态。收到 SIGTERM 后，将 readiness 设为 false，等待短暂的状态传播时间，然后通过 aiohttp cleanup 在限定的关闭超时内完成清理。
5. 将部署的 readiness 检查从 `/health` 改为 `/ready`；`/health` 继续用作 liveness。
6. 获得用户明确授权后，再次运行相同的重点测试。

### 任务 4：文档与发布

**涉及文件：**
- 修改：`../NexusAi-Dispatcher/README.md`
- 修改：`docs/designs/2026-08-13-dispatcher-inference-separation.md`

1. 记录正常路由、容量响应即时重试、基础设施故障退避、熔断阈值、readiness 摘流，以及保留 20%～30% 容量余量的建议。
2. 明确 Service 负载均衡不等于精确选择空闲 Worker，并且仍必须根据 `taskId` 去重。
3. 检查暂存区差异；只有在获得明确授权并完成重点验证后，才分别提交和推送两个仓库。
