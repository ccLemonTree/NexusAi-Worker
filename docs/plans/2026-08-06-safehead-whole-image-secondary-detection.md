# Safehead 全图二次检测实施计划

> **给 Claude：** 实施时必须使用子技能 `superpowers:executing-plans`，逐项执行本计划。

**目标：** 对整图只执行一次第二标签推理；仅当第一标签的 Safehead 检测框与所有第二标签检测框既不相交也不接近时，才返回该检测框。

**架构：** 保留现有 `Model.execute` 入口，但将检测框邻近规则拆成一个小型纯函数。存在第一标签结果时，对整图执行一次已配置的第二标签推理；将每个第一标签检测框按自身宽高向四周扩展 20%，再与所有第二标签检测框比较并完成过滤。

**技术栈：** Python、pytest、现有 `BoundingBox` 对象和 `analyseRun` 推理 API。

---

## 需求理解

- `self.logicResult` 包含第一标签（`personnew`）的检测结果。
- 第二标签从 `cfg.logicModelDict[self.logicModelName][1]["label"]` 读取，当前配置为 `yolov5nohead`。
- 只有至少存在一个第一标签结果时，才执行第二标签推理。
- 第二标签推理接收整张图片，每次 `execute` 调用只执行一次。
- 如果第二标签检测框与第一标签检测框相交，或与第一标签检测框向四周扩展 20% 后的区域相交，则认为两者相关。
- 只返回与所有第二标签检测框都不相关的第一标签检测框。
- 模型配置和其他流水线不在本次改动范围内。

## 前提假设

- “扩展 20%”表示左右两侧各增加第一标签检测框宽度的 20%，上下两侧各增加其高度的 20%。
- 边界接触视为相交或接近。
- 如果第二标签推理没有返回检测框，则保留所有第一标签检测框。
- 返回的检测框保持原始坐标，只修改 `classname`，与当前行为一致。

## 决策记录

- 使用按比例扩展而不是固定像素阈值，使判断能适应不同图片和目标尺寸。
- 只扩展第一标签检测框，因为它是当前待判断的候选框，并能提供稳定的参考尺寸。
- 使用轴对齐矩形重叠判断而不是中心点距离，以适配尺寸差异较大的检测框。
- 将一次整图推理放在第一标签结果循环之外，避免重复推理。

### 任务 1：几何逻辑回归测试

**涉及文件：**
- 创建：`tests/test_safehead.py`
- 测试：`tests/test_safehead.py`

**步骤 1：编写预期失败的测试**

针对直接重叠、间距位于 20% 扩展范围内、间距超出扩展范围、边界接触和第二标签列表为空等场景添加重点测试。

**步骤 2：运行测试并确认失败**

运行：`python -m pytest tests/test_safehead.py -v`

预期结果：失败，因为 Safehead 邻近/过滤辅助函数尚未实现。

### 任务 2：实现整图过滤

**涉及文件：**
- 修改：`api/infer/Model_pipline/Safehead/Safehead.py`
- 测试：`tests/test_safehead.py`

**步骤 1：添加最小化几何辅助函数**

实现轴对齐重叠判断，将第一标签检测框按默认比例 `0.2` 扩展后再比较。

**步骤 2：更新 `Model.execute`**

没有第一标签结果时直接返回。否则使用 `[self.picture, self.picture]` 调用一次 `analyseRun`，然后只保留扩展区域未与任何第二标签检测框重叠的第一标签检测框。

**步骤 3：运行测试并确认通过**

运行：`python -m pytest tests/test_safehead.py -v`

预期结果：通过。

### 任务 3：仓库级验证

**涉及文件：**
- 验证：`api/infer/Model_pipline/Safehead/Safehead.py`
- 验证：`tests/test_safehead.py`

**步骤 1：执行语法检查**

运行：`python -m compileall kafka api tools utils`

预期结果：退出码为 0。

**步骤 2：重新运行重点测试**

运行：`python -m pytest tests/test_safehead.py -v`

预期结果：全部测试通过，无失败项。
