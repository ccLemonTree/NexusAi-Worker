# Triton 推理压力测试工具

## 功能说明

测试 Triton 推理服务的吞吐量，支持两种测试场景：

1. **测试所有 5 个标签**（含 VLM 大模型）：
   - Smoking-handsmoking (yolov11det_smoking)
   - 26_smoke (yolov26det_fire)
   - Hot-work (yolov26det_fire - 逻辑标签)
   - 26_fire (yolov26det_fire)
   - VLM 大模型推理

2. **仅测试 4 个小模型标签**（排除 VLM）：
   - 只调用 Triton 小模型，跳过 VLM 大模型推理

## 使用方法

### 1. 准备测试环境

```bash
# 登录正式环境服务器
ssh root@36.140.131.182
# 密码: w3X)yJ!G6p

# 准备测试图片（使用任意一张图片）
# 例如从容器挂载路径复制一张
cp /ai/capture/back/search_pic/xxx.jpg /tmp/test.jpg
```

### 2. 运行测试

#### 方式 1: 持续时间测试（推荐）

测试 10 秒内能处理多少张图片：

```bash
# 测试所有 5 个标签（含 VLM）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 16 --labels all

# 仅测试 4 个小模型标签（排除 VLM）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 16 --labels small
```

#### 方式 2: 固定次数测试

推理 100 次，看需要多长时间：

```bash
# 测试所有 5 个标签
python stress_test.py --image /tmp/test.jpg --count 100 --concurrent 16 --labels all

# 仅测试小模型
python stress_test.py --image /tmp/test.jpg --count 100 --concurrent 16 --labels small
```

### 3. 调整并发数测试

```bash
# 低并发（4 并发）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 4 --labels small

# 中并发（8 并发）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 8 --labels small

# 高并发（16 并发，当前 Kafka 配置）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 16 --labels small

# 更高并发（32 并发，用于测试极限）
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 32 --labels small
```

## 输出示例

```
================================================================================
持续时间压力测试
================================================================================
测试图片: /tmp/test.jpg
测试时长: 10 秒
并发数: 16
测试标签: 仅4个小模型标签
================================================================================

✓ 图片读取成功: (1080, 1920, 3)

进度: 10 次推理完成, 成功 10, 当前QPS: 0.87
进度: 20 次推理完成, 成功 20, 当前QPS: 1.23
进度: 30 次推理完成, 成功 30, 当前QPS: 1.45

================================================================================
测试结果统计
================================================================================
总耗时: 10.234 秒
总请求数: 35
成功: 35 (100.0%)
失败: 0 (0.0%)

吞吐量:
  QPS (每秒推理图片数): 3.42

延迟统计 (秒):
  最小值: 0.802
  最大值: 12.456
  平均值: 4.523
  中位数: 3.891
  P95: 10.234
  P99: 11.987
================================================================================
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--image` | 测试图片路径（必需） | `/tmp/test.jpg` |
| `--duration` | 持续测试时长（秒） | `10` |
| `--count` | 推理次数（与 duration 二选一） | `100` |
| `--concurrent` | 并发数（默认 16） | `8` |
| `--labels` | 测试标签：`all`=所有 5 个标签（含 VLM），`small`=仅 4 个小模型 | `all` 或 `small` |

## 关键指标说明

### QPS (每秒推理图片数)
- **定义**: 每秒能处理多少张图片
- **计算**: 总请求数 / 总耗时
- **目标**: 越高越好

### 延迟统计
- **最小值**: 最快的单次推理时间（理想情况下的延迟）
- **平均值**: 所有推理的平均耗时
- **P95/P99**: 95%/99% 的请求在此时间内完成
- **最大值**: 最慢的单次推理时间

## 测试建议

### 1. 先测试小模型（排除 VLM）

```bash
# 快速测试 10 秒
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 16 --labels small
```

**预期结果**：
- 如果 QPS < 2：说明 GPU 排队严重，考虑降低并发或增加 Triton 实例数
- 如果 QPS 2-5：正常范围，取决于模型复杂度
- 如果 QPS > 5：性能良好

### 2. 再测试包含 VLM

```bash
python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent 16 --labels all
```

**预期结果**：
- VLM 会显著降低 QPS（大模型推理耗时 3-10 秒）
- 观察 VLM 对整体吞吐量的影响

### 3. 测试不同并发数

```bash
# 测试 4, 8, 16, 32 并发
for c in 4 8 16 32; do
    echo "测试并发数: $c"
    python stress_test.py --image /tmp/test.jpg --duration 10 --concurrent $c --labels small
done
```

找到最优并发数（QPS 最高且延迟稳定）

## 在容器内运行

如果需要在运行中的容器内测试：

```bash
# 1. 将脚本复制到容器
docker cp stress_test.py <container_id>:/app/

# 2. 进入容器
docker exec -it <container_id> bash

# 3. 准备测试图片（使用挂载路径）
ls /ai/capture/back/search_pic/*.jpg | head -1
# 复制路径，例如: /ai/capture/back/search_pic/D1783502816221/xxx.jpg

# 4. 运行测试
python stress_test.py --image <图片路径> --duration 10 --concurrent 16 --labels small
```

## 故障排查

### 问题 1: 导入错误

```
ModuleNotFoundError: No module named 'kafka'
```

**解决方案**: 确保在项目根目录运行，或设置 PYTHONPATH：

```bash
export PYTHONPATH=/app:$PYTHONPATH
python stress_test.py --image test.jpg --duration 10
```

### 问题 2: 图片不存在

```
❌ 错误：图片文件不存在: test.jpg
```

**解决方案**: 使用绝对路径或确认文件存在：

```bash
ls -lh /tmp/test.jpg
python stress_test.py --image /tmp/test.jpg --duration 10
```

### 问题 3: 大量超时错误

如果看到大量 "TimeoutError" 或 "CONNECTION_ERROR"：

1. 检查 Triton 服务是否正常：
   ```bash
   curl http://10.120.0.66:40008/v2/health/ready
   ```

2. 检查网络连通性
3. 降低并发数重试

## 性能优化建议

根据测试结果：

### 如果 QPS 低且延迟高（大部分请求 > 5 秒）

**原因**: Triton GPU 排队严重

**解决方案**:
1. 增加 Triton 模型实例数（修改 config.pbtxt 的 `instance_group.count`）
2. 降低 Kafka 并发数 `KAFKA_MAX_CONCURRENT`
3. 启用动态批处理（如果模型支持）

### 如果 QPS 达标但 P99 延迟很高

**原因**: 偶尔出现排队，但大部分请求正常

**解决方案**:
1. 增加 1-2 个 Triton 实例
2. 监控 GPU 利用率，确保不是 GPU 性能瓶颈

### 如果不同并发数 QPS 相近

**原因**: 已达到 Triton 处理上限

**解决方案**:
1. 使用更低的并发数（减少排队时间）
2. 横向扩展 Triton 服务
