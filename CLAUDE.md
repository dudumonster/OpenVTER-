# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 常用命令

```bash
# 本地开发（Windows / Mac）
python src/train.py --config configs/local_windows.yaml      # Windows
python3 src/train.py --config configs/local_mac.yaml          # Mac

# 服务器运行（从 YAML 配置处理单个视频）
python3 src/train.py --config configs/server.yaml

# 服务器批量运行（处理场景目录下所有视频）
bash scripts/run_scene_server.sh <场景目录> <道路配置目录>

# 原始入口（仍然可用）
python video_inference_main.py -c <config.json> -s 3          # step 3 = 完整流水线
python video_inference_main.py -c <config.json> -s 3 -m       # 多进程模式

# 服务器上激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate openvter
```

## 架构概览

**OpenVTER** 从无人机航拍视频中提取车辆轨迹，已发表于 IEEE T-ITS (2024)。流水线为：视频稳定化 → 图像切分 → 旋转目标检测 → 多目标跟踪 → 轨迹后处理。

### 入口点

- **`src/train.py`** — 统一启动器。虽然名为 train，但实际标准化的是 `video_inference` 流程（而非训练）。读取 YAML 配置 → 解析 `${var}` 变量插值和相对路径 → 写入运行时 JSON → 调用 `video_inference_main.run_pipeline()`。
- **`src/run_scene.py`** — 场景批量运行器。使用 `configs/server.yaml` 作为模板，为每个视频生成独立配置（填入 `video_file`、`road_config`、`save_folder` 等），然后逐个调用 `train.run_config()`。

### 配置系统（双层）

1. **YAML 配置**（`configs/*.yaml`）— 人工编辑。支持 `${variable}` 变量插值，递归解析。`train.py` 中的 `PATH_KEYS` 会自动解析为绝对路径。
2. **JSON 配置**（`config/demo_config/**/*.json`）— 旧格式，由 `video_process.py` 消费。YAML 层在运行时会生成这些 JSON。

关键配置字段：`pipeline`（`"stab"`、`"det"` 组成的列表）、`detection`（模型配置列表）、`tracking`（跟踪器配置列表）、`out_fps`、`inference_batch_size`、`subsize_height/width`、`split_gap`。

### 检测模块（`detection/VehicleDetModule.py`）

工厂模式，根据 `model_name` 分发：
- `centernet_bbavectors` — TorchScript 模型，通过 BBA 向量输出旋转框，输入尺寸 640×512
- `yolox_r` — 带旋转输出的 YOLOX
- `yolov5` — 标准 YOLOv5（用于 VRU 类别：行人、自行车等）
- `mmrotate` — MMRotate 封装

每个模型必须实现 `det_images_batch(images_ls)`，返回 numpy 数组列表（每张图一个），形状为 `[N, 10]` = `[x1..y4, score, class]`。

### 跟踪模块（`tracking/sort_r/sort_r.py`）

针对旋转边界框修改的 SORT 算法。使用 mmcv 的 `box_iou_rotated`（GPU）计算 IoU 矩阵，然后在 CPU 上运行匈牙利算法（`lap.lapjv` 或 `scipy.linear_sum_assignment`）。卡尔曼滤波器状态为 `[cx, cy, w, h, theta, v_cx, v_cy, v_scale]`。

### 核心流水线（`video_inference/video_process.py`）

`DroneVideoProcess._process_img()` 每帧执行：
1. 稳定化（通过预计算的仿射变换矩阵进行仿射变换，可选）
2. 掩膜应用（与道路多边形做 bitwise_and）
3. 图像切分（通过 `splitbase` 做有重叠的滑动窗口，默认 gap=160px）
4. **对每批 tile → 对每个检测模型**：GPU 推理 → 每个 tile 的 NMS → 位置偏移补偿
5. 跨所有 tile 结果的全局 NMS
6. 跟踪更新（SORT-R）
7. 坐标变换（通过仿射矩阵将像素坐标转为世界坐标）
8. 车道分配（点在多边形内判断）
9. 类别平滑（滑动窗口多数投票 + 基于车长的重分类）
10. 绘图（旋转框 + 标签）
11. 视频输出（mp4v 编码）

### 图像切分（`utils/VideoTool.py`）

`splitbase` 类：有 `gap` 重叠的滑动窗口。宽度步长 = `subsize_width - gap`，高度步长 = `subsize_height - gap`。首次调用 `split_image()` 计算所有切分位置（跳过黑色占比 >98% 的 tile）；后续帧使用 `split_image_with_position()` 直接按位置切分以加速。

## 性能注意事项

默认 `server.yaml` 使用**两个检测模型**（CenterNet-BBA 检测 5 类车辆 + YOLOv5 检测 6 类 VRU），且 `out_fps: 29.97`（处理每一帧）。在 A100 上，一个 5 分钟视频（约 9000 帧）耗时约 140 分钟。主要瓶颈：

1. 两个模型均以 FP32 运行。在 A100 上切换到 FP16/AMP 可获得 1.5-2 倍加速。
2. `VehicleDetModule._nms_rotated_tensor()` 中的每 tile NMS 与 `_process_img()` 中的全局 NMS 重复。去除后可节省约 8-12%。
3. `cv2.minAreaRect()` 在 Python for 循环中对每个检测框逐个调用（CPU）。在 GPU 上向量化处理会有效果。
4. 每个 tile 存在多次 CPU↔GPU 往返（`numpy → torch → GPU → numpy → torch → GPU → numpy`）。
5. 绘图（`draw_oriented_bboxs`）无论 `output_video` 设置为何都会执行。

## 关键模式

- **配置驱动**：流水线行为、模型选择、类别映射、颜色方案均在 JSON/YAML 中配置。
- **工厂模式**：`VehicleDetModule` 和 `VehicleTrackingModule` 根据 `model_name` 实例化对应后端。
- **多模型融合**：`detection` 和 `tracking` 配置字段接受列表。多个检测器的结果合并；多个跟踪器通过 `id_offset` 避免 ID 冲突，分别处理不同的类别组。
- **坐标流水线**：像素坐标 → 通过 `pixel2xy_matrix`（道路配置中的仿射变换矩阵）转为世界坐标。车辆长度（米）由世界坐标或 `length_per_pixel` 回退方案计算得出。

## 运行环境

Conda 环境来自 `DMS-full.yml`（Python 3.9, PyTorch 1.12.0+cu113, mmcv-full, opencv, shapely, streamlit）。实际使用需要 GPU。模型权重存放在 `checkpoints/`（gitignore 忽略，需单独下载）。

## 行为准则

旨在减少常见 LLM 编码错误的行为指南。请根据项目需要与项目特定说明合并使用。

**权衡取舍：** 以下准则偏向谨慎而非速度。对于简单任务，自行判断即可。

### 1. 先思考，后编码

**不要假设。不要隐藏困惑。有取舍时主动暴露。**

实现之前：
- 明确陈述你的假设。如果不确定，就提问。
- 如果存在多种理解方式，全部列出来——不要默默选一个。
- 如果有更简单的方案，直接说出来。必要时可以据理力争。
- 如果有不清楚的地方，停下来。说出你困惑的点。提问。

### 2. 简单至上

**用最少的代码解决问题。不写任何推测性代码。**

- 不添加超出需求范围的功能。
- 不为仅使用一次的代码创建抽象。
- 不添加未被要求的"灵活性"或"可配置性"。
- 不为不可能发生的场景添加错误处理。
- 如果你写了 200 行代码但 50 行就够了，重写它。

问自己："资深工程师会认为这是过度复杂吗？" 如果是，就简化。

### 3. 精准手术式修改

**只动你必须改的。只清理你自己造成的混乱。**

编辑已有代码时：
- 不要"顺手优化"相邻代码、注释或格式。
- 不要重构没坏的东西。
- 匹配现有代码风格，即使你更倾向于另一种写法。
- 如果你注意到无关的死代码，提一下它——但不要删除它。

当你的修改产生了孤立的残余时：
- 删除因 **你的修改** 而不再使用的 import、变量、函数。
- 除非被要求，否则不要删除之前就存在的死代码。

检验标准：每一行改动都应该能直接追溯到用户的需求。

### 4. 目标驱动的执行

**定义成功标准。循环验证直到通过。**

将任务转化为可验证的目标：
- "添加校验" → "为无效输入编写测试，然后让它们通过"
- "修复 bug" → "编写一个能复现 bug 的测试，然后让它通过"
- "重构 X" → "确保重构前后测试都通过"

对于多步骤任务，给出简要计划：
1. [步骤] → 验证：[检查项]
2. [步骤] → 验证：[检查项]
3. [步骤] → 验证：[检查项]

强大的成功标准让你能独立地循环迭代。
模糊的标准（"让它能工作就行"）则需要不断地反复澄清。

---

**这些准则正在生效的标志：** diff 中不必要的改动变少、因过度复杂而被要求返工的情况减少、澄清性的问题在实现之前被提出而非在犯错之后。

## AI 协作研发记录工作流

本项目采用 Codex、Claude Code、Cursor、Gemini CLI、人工开发者以及其他 AI Agent 均可参与的协作方式。无论修改来自哪个工具或人员，**Git diff 都是变更分析的事实依据**。

### 角色与原则

- Claude Code 默认负责变更审查、研发记录整理、项目记忆维护和飞书同步。
- Claude Code 也可以参与开发，但开发完成后的记录流程仍然适用。
- Codex、Cursor、Gemini CLI、人工开发者或其他 Agent 产生的修改，都应被纳入同一套记录流程。
- 不自动提交 Git commit、不自动 push，除非用户明确要求。
- 不把大量临时调试过程写入源码注释；源码注释只保留稳定接口、字段含义、长期有效的逻辑说明。
- 项目级长期规则、架构约定、数据格式说明、历史决策原因可以持续维护在 `CLAUDE.md` 和 `AGENTS.md` 中。

### 研发记录文件

- `.ai-worklog/pending.md`：临时记录区。用于保存任务目标、修改原因、关键变量、输出 shape、数据结构变化、字段变化、网络结构变化、调试结论、测试命令、测试结果、风险点和待确认问题。
- `.ai-worklog/latest.md`：本轮正式工作记录。由 `/worklog` 或同类记录任务生成。
- `docs/dev-log.md`：长期研发日志。每次记录任务完成后，将 `.ai-worklog/latest.md` 追加到这里。
- `.env.local`：本地私有配置，例如飞书机器人 Webhook。此文件不提交 Git。

## `/worklog` 命令约定

Claude Code 执行 `/worklog` 时应默认只做记录和分析，不主动修改业务代码。记录流程：

1. 读取 `git status --short`、`git diff --stat`、`git diff --name-only`。
2. 必要时读取关键文件的 `git diff` 内容。
3. 读取 `.ai-worklog/pending.md` 中由 Codex、Cursor、人工开发者或其他 Agent 补充的临时信息。
4. 生成 `.ai-worklog/latest.md`。
5. 追加 `docs/dev-log.md`。
6. 飞书文档同步：
   - 使用 feishu_mcp.py 的 `append` 模式将本轮日志追加到飞书文档，保留用户已有的手写内容
   - 飞书文档 ID 配置在 `.env.local` 的 `FEISHU_DOC_ID` 字段
   - 使用 UAT（用户访问令牌）以用户身份操作文档，确保用户有编辑权限
   - 每轮日志前加 `---` 分隔线，便于区分不同轮次
7. 通过群聊机器人 Webhook 发送简短通知（文档链接 + 本轮摘要）。
8. 向用户汇报记录文件位置、飞书文档状态、群聊通知状态、仍需人工确认的问题。

### 记录内容要求

每轮记录应尽量覆盖：

- 本轮任务目标
- 涉及模块和文件
- 核心代码变化
- 新增或修改的数据结构
- 字段名称、字段含义、字段顺序变化
- 网络结构变化
- 张量 shape / 输出维度变化
- 关键中间变量
- 调试输出和临时判断
- 影响范围和可能受影响的下游脚本
- 已运行测试、测试命令和测试结果
- 风险点
- 建议下一步测试命令
- 需要人工确认的问题
