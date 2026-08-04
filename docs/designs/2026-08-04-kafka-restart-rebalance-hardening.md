# Kafka Worker 重启与 Rebalance 最小加固设计

## 背景与目标

生产环境包含 7 台服务器、每台 4 个 worker，共 28 个动态消费者；`model_analyse` 有 40 个 partition，输入约 101 条/秒，结果约 108 条/秒。Broker 不支持 KIP-345。Compose 重启会触发全组 rebalance，旧 generation 的在途任务提交 offset 时出现 `IllegalGenerationError` 和 `UnknownMemberIdError`。

本次保留线上每 worker 16 并发及逐消息 commit 模型，不重构为分区串行队列或连续 offset 跟踪器。目标仅是缩短重启扰动、阻止明确过期的 commit，并改善优雅退出。

## 语义与约束

- 正常运行期间维持现有吞吐与处理流程。
- 允许重启窗口少量重复，不因旧 generation commit 造成额外丢失。
- worker 进程首次获得 assignment 时主动跳到 partition 末尾并 commit，明确丢弃启动前 backlog。
- 后续普通 rebalance 从已提交 offset 继续，不再 seek-to-end。
- 不修改消息结构、推理逻辑、结果 Topic、认证或下游接口。

## 方案比较

1. 分区内串行：语义简单，但 40 个 partition 理论并发仅 40，难以稳妥支撑包含 VLM 的 101 QPS。
2. 连续 offset 跟踪：保留吞吐且可避免越级提交，但状态和测试复杂度高。
3. 线上模型加最小生命周期保护（采用）：保持当前吞吐，仅处理 revoke、旧 epoch commit 和 SIGTERM。

## 最终设计

### Rebalance epoch

新增共享 rebalance 状态，包含 `epoch`、`rebalancing` 和状态锁。每个 `_handle_one()` 创建时捕获当前 epoch，并传给 `_safe_commit()`。

`ConsumerRebalanceListener.on_partitions_revoked()` 首先设置 `rebalancing=True` 并增加 epoch，使旧任务失效；随后停止调度新消息，最多等待 30 秒让在途任务结束。旧任务允许完成推理和结果发送，但 epoch 不匹配时跳过 commit。超时任务不提交，由新 owner 重复处理。

`on_partitions_assigned()` 完成必要的启动初始化后清除 rebalance 状态。正常任务仍按现有方式逐条 commit。已经发往 Broker 的 commit 仍可能返回一次 generation 错误，继续由现有异常处理降级为 warning。

### 首次启动跳过 backlog

当 `KAFKA_SKIP_BACKLOG_ON_START=true` 时，进程第一次获得稳定 assignment 后执行 `seek_to_end()`，读取各 partition 的当前位置并 commit。只有全部 commit 成功后才开始调度消息。若初始化期间再次 rebalance，则放弃本轮并在下一次 assignment 重试。普通 rebalance 不执行该逻辑。

### 优雅退出

将无限 `async for` 改为带超时的 `consumer.getmany()` 循环，使空闲消费者也能定期检查 `_shutdown_event`。收到 SIGTERM 后停止拉取新消息，最多等待 `KAFKA_SHUTDOWN_DRAIN_TIMEOUT=90` 秒；随后停止 consumer 和 producer。Compose 设置 `stop_grace_period: 2m`，并采用逐 worker 滚动重启。

## 配置

```yaml
KAFKA_REBALANCE_DRAIN_TIMEOUT: "30"
KAFKA_SHUTDOWN_DRAIN_TIMEOUT: "90"
KAFKA_SKIP_BACKLOG_ON_START: "true"
stop_grace_period: 2m
```

## 测试策略

- 正常 epoch 可以 commit；旧 epoch 不调用 Broker commit。
- revoke/assign 后新任务恢复提交，旧任务只允许重复。
- 空闲消费者收到 SIGTERM 后 1–2 秒内退出拉取循环。
- 首次 assignment seek-to-end 并 commit；普通 rebalance 不跳末尾。
- 启动 commit 失败时不得开始消费。
- drain 超时后不提交旧 offset。
- Compose 集成测试逐个重启 worker，验证新消息继续处理、容器稳定且无 rebalance 循环。

## 决策记录

1. 保留线上并发模型，因为 40 partition 串行方案对 101 QPS 缺少余量。
2. 不实现连续 offset 跟踪器，控制本次变更范围与生产风险。
3. 使用本地 epoch 阻止明确过期的 commit，同时接受少量重复。
4. 每个进程仅首次 assignment 跳过 backlog，普通 rebalance 不丢历史消息。
5. Broker 升级前不恢复静态成员；部署采用逐 worker 滚动重启。
