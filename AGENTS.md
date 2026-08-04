# 仓库贡献指南

## 项目结构与模块组织

`kafka_main.py` 是工作进程入口。Kafka 消息处理位于 `kafka/`；配置、日志、HTTP 和并发工具集中在 `tools/` 与 `utils/`。推理编排位于 `api/infer/`，其中 `Model_pipline/` 存放模型流水线，`Triton_model/` 存放 Triton 客户端及后处理逻辑。其他 API 按功能放在 `api/` 对应子目录。性能与集成诊断脚本位于 `tests/performance/`，示例图片位于 `example/`。

## 构建、测试与开发命令

- `python -m venv .venv`：创建本地虚拟环境。
- `python -m pip install -r requirements.txt`：安装锁定版本的运行时依赖。
- `python kafka_main.py`：使用 `.env` 或注入的环境变量启动工作进程；运行前须确保 Kafka、Triton、Milvus 和对象存储可访问。
- `python -m compileall kafka api tools utils`：快速检查 Python 语法。
- `python tests/performance/test_e2e.py --help`：查看端到端诊断脚本参数。
- `docker build -t nexusai-worker .`：构建 Python 3.12 工作进程镜像。
- `docker compose -f docker-compose-stats.yml config`：部署前校验 Compose 配置及变量替换。

## 编码风格与命名约定

使用四空格缩进并遵循 PEP 8。模块、函数和变量使用 `snake_case`；新建类使用 `PascalCase`；常量及环境变量使用 `UPPER_SNAKE_CASE`。公共接口和并发敏感代码应添加类型标注，异步 I/O 不得执行阻塞操作。模型目录名称可能被配置引用，请勿随意修改。仓库尚未统一配置格式化工具，避免提交无关的格式调整。

## 测试规范

现有测试多为依赖外部服务的性能诊断脚本，而非隔离的单元测试。诊断脚本命名为 `test_<行为>.py` 并放入 `tests/performance/`；可独立运行的单元测试应放在 `tests/`。注明所需端点、模型、测试数据和预期指标。提交前执行语法检查及所有相关诊断，并在 PR 中记录完整命令和结果。

## 提交与拉取请求规范

提交历史主要使用简短的祈使句，例如 `Fix negative bbox width/height` 和 `Add partition assignment logging`。每个提交只处理一个明确问题，并在正文说明运行或部署影响。PR 应概述行为变化、列出配置或模型依赖、关联相关 Issue，并提供测试证据。推理或并发改动应附关键日志及前后性能数据；仅在视觉输出变化时提供截图。

## 安全与配置

将 `.env*`、消息代理凭据、对象存储密钥及内部端点视为敏感信息。不得在 Compose 文件或源码中新增明文凭据；应通过环境变量或部署平台的 Secret 注入。提交日志、Issue 和测试产物前必须移除敏感值。
