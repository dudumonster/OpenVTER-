# OpenVTER 弱势交通参与者 OBB 伪标注方案 v2 — 实验计划

**创建日期**：2026-06-25
**实验负责人**：待指定
**Codex 协助**：视觉评估任务标记为 [Codex]

---

## 1. 实验目标

对比三种弱监督 OBB 伪标注方案在 VisDrone 弱势交通参与者类别上的效果：

| 方案 | 代号 | 定位 | 分割 | 语义审核 |
| --- | --- | --- | --- | --- |
| A | HBB + SAM2 | HBB expanded box | SAM2 box-prompt | 无 |
| B | HBB + Grounded-SAM-2 | GroundingDINO text prompt + SAM2 box prompt | SAM2 | GroundingDINO 文本匹配 |
| C | HBB + Grounded-SAM-2 + VLM Review | 同 B | SAM2 | Qwen/Gemini 二次语义审核 |

## 2. 实验范围

### 2.1 数据选择

- **数据来源**：VisDrone2019-DET
- **优先 split**：val（548 张图），作为首批实验
- **后续 split**：train（6471 张图），在 val 验证后运行
- **不包含**：test-dev（无公开标注，暂不使用）

### 2.2 类别和样本

- **目标类别**（4 类）：
  - bicycle (class_id=1)
  - motor (class_id=2)
  - tricycle (class_id=3)
  - awning_tricycle (class_id=4)
- **每类采样数**：100 个样本（方案 A/B/C 各使用同一组样本）
- **采样策略**：
  - 从 val 中按 class_id 随机选取
  - 确保覆盖不同尺寸（小 <50px、中 50-150px、大 >150px）
  - 确保覆盖不同遮挡等级（0: 无遮挡, 1: 部分遮挡, 2: 严重遮挡）
  - 如果 val 中某类不足 100 个，从 train 中随机补充

### 2.3 VisDrone val 实际可用样本量

| 类别 | val 总样本数 | 预计可抽 100 个？ |
| --- | --- | --- |
| bicycle | 1,287 | 是 |
| motor | 4,886 | 是 |
| tricycle | 1,045 | 是 |
| awning_tricycle | 532 | 是 |

val 中所有 4 类样本均充足，不需要从 train 补充。

## 3. 评估指标

### 3.1 自动指标（可由代码计算）

| 指标 | 计算方式 | 含义 |
| --- | --- | --- |
| 自动通过率 | `auto_accept / total` | 不需要人工审核的比例 |
| mask 空结果比例 | `mask_empty / total` | 分割完全失败的比例 |
| 异常面积比例 | `area_ratio_out_of_range / total` | OBB 面积异常的比例 |
| 中心偏移异常比例 | `center_shift > 0.40 / total` | OBB 中心偏移过大的比例 |
| 平均 final_score | `mean(final_score)` | 整体质量分均值 |
| 平均 geometry_score | `mean(geometry_score)` | 几何质量分均值 |
| 平均 semantic_score | `mean(semantic_score)` | 语义质量分均值 |
| VLM-HBB 类别一致率 | `vlm_class_agrees_with_hbb / total_with_vlm` | VLM 判断与 VisDrone HBB 标签的一致率 |
| 平均处理时间 | `total_time / total_samples` | 每样本耗时（含模型推理） |

### 3.2 人工指标（需 [Codex] 视觉评估）

| 指标 | 评估方式 | 含义 |
| --- | --- | --- |
| 人工审核通过率 | 人工逐张审核 auto_accept 样本 | auto_accept 样本中真正可用的比例 |
| OBB 贴合度合格率 | 人工判断 OBB 四点是否贴合目标 | 合格样本占比 |
| 类别正确率 | 人工判断 OBB 类别是否正确 | 分类正确率 |
| mask 分割质量 | 人工判断 mask 是否完整覆盖目标 | 分级：好 / 可接受 / 差 |

### 3.3 成本指标

| 指标 | 计算方式 |
| --- | --- |
| 每样本 GPU 时间 | 端到端处理时间（含 GroundingDINO + SAM2） |
| 每样本 VLM API 调用成本 | 仅方案 C：Qwen/Gemini API 调用费用估算 |
| 人工审核节省率 | `(total - review_queue) / total`，即自动入库比例 |

## 4. 实验步骤

### Step 0: 环境准备（本周）

- [ ] 确认 GroundingDINO 权重和配置文件可用
- [ ] 确认 SAM2 权重可用（已有 v1 环境可复用）
- [ ] 创建 conda 环境 `openvter-obb-vlm`（或扩展现有 `openvter-obb-sam`）
- [ ] 安装 GroundingDINO 依赖
- [ ] （可选）安装 Florence-2 作为备选后端

### Step 1: 方案 A（baseline）运行（1 天）

```bash
# 从 val 中每类采样 100 个，运行 v1 流水线（等效于 v2 只开 SAM2 后端）
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate \
  --splits val \
  --segmenter sam2 \
  --sam-checkpoint checkpoints/sam2/sam2.1_hiera_base_plus.pt \
  --sam2-config sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --expand-ratio 0.20 \
  --pseudo-root dataset_construction/derived/visdrone_pseudo_obb_v2_expA \
  --yolo-root dataset_construction/derived/visdrone_yolo_obb_v2_expA \
  --save-mask-crops --copy-mode symlink
```

**注意**: 因为 v1 脚本没有采样功能，需要先过滤 quality.jsonl 获取每类 100 个 sample。可以在生成后通过 sample_id 过滤。如果后续实现了 v2 脚本，直接用 `--experiment-mode --sample-per-class 100`。

**产出**：
- `quality.jsonl`：所有样本的质量记录
- `review_queue.jsonl`：待审核队列
- YOLO-OBB 标签（auto_accept 样本）

### Step 2: 方案 B（Grounded-SAM-2）运行（1-2 天）

需要通过 v2 脚本（待实现）运行 Grounded-SAM-2 流水线。

**产出**：同方案 A，额外包含 VLM 字段（vlm_box_xyxy, vlm_class_name, vlm_class_confidence 等）。

### Step 3: 方案 C（+VLM Review）运行（1-2 天）

在方案 B 基础上，对 review 队列样本调用 Qwen/Gemini 做二次语义审核。

**产出**：同方案 B，额外包含 VLM review 结果。

### Step 4: 自动指标汇总（1 小时）

运行分析脚本，输出方案 A/B/C 对比表。

### Step 5: [Codex] 人工视觉评估（2-3 天）

- 从每个方案的 auto_accept 中每类抽取 30 个样本，生成对比视图。
- 从 review 队列中每类抽取 30 个样本，判断是否有可挽救的。
- 人工评定 OBB 贴合度和类别正确率。

### Step 6: 汇总和决策（1 天）

根据实验结果决定：
- 推荐后端组合（是否用 Grounding，是否用 VLM review）
- 全量 train 运行参数
- 是否需要调整 prompt 模板

## 5. 结果记录模板

### 5.1 自动指标对比表

| 指标 | 方案 A (SAM2 only) | 方案 B (Grounded-SAM-2) | 方案 C (+VLM Review) |
| --- | --- | --- | --- |
| 总样本数 | 400 | 400 | 未运行 |
| **bicycle** | | | |
| — auto_accept | 21 | 16 | |
| — review | 78 | 84 | |
| — reject | 1 | 0 | |
| — mask_empty | 0 | 0 | |
| — 平均 final_score | 0.5836 | 0.5831 | |
| **motor** | | | |
| — auto_accept | 26 | 14 | |
| — review | 73 | 86 | |
| — reject | 1 | 0 | |
| — mask_empty | 0 | 0 | |
| — 平均 final_score | 0.6201 | 0.6124 | |
| **tricycle** | | | |
| — auto_accept | 53 | 46 | |
| — review | 47 | 54 | |
| — reject | 0 | 0 | |
| — mask_empty | 0 | 0 | |
| — 平均 final_score | 0.6948 | 0.6596 | |
| **awning_tricycle** | | | |
| — auto_accept | 49 | 35 | |
| — review | 51 | 65 | |
| — reject | 0 | 0 | |
| — mask_empty | 0 | 0 | |
| — 平均 final_score | 0.6865 | 0.6457 | |
| **全部** | | | |
| — 自动通过率 | 149/400 = 37.25% | 111/400 = 27.75% | |
| — 平均处理时间 | 未计时（约 2-3 min / 400，MPS） | 未计时（约 27 min / 400，GroundingDINO C++ ops CPU fallback） | |
| — VLM-HBB 一致率 | N/A | 376/376 = 100%（有 VLM signal 的样本） | |

### 5.2 [Codex] 人工评估对比表

| 指标 | 方案 A | 方案 B | 方案 C |
| --- | --- | --- | --- |
| OBB 贴合合格率 (auto_accept) | contact sheet 初筛：三轮/带篷三轮较稳，bicycle/motor 小目标风险较高；待逐图复核 | contact sheet 初筛：更保守，部分小目标框更贴合，但自动接收数更少；待逐图复核 | |
| OBB 贴合合格率 (review) | contact sheet 初筛：review 中仍有可救样本，也有明显框过大/多目标粘连；待逐图复核 | contact sheet 初筛：review 增多，需逐图判断是否为合理降级或过度保守 | |
| 类别正确率 (auto_accept) | 未逐图统计 | | |
| mask 分割质量 - 好 % | 未逐图统计 | | |
| mask 分割质量 - 可接受 % | 未逐图统计 | | |
| mask 分割质量 - 差 % | 未逐图统计 | | |
| 人工审核通过率 | 未逐图统计 | | |

## 6. 成功标准

| 等级 | 标准 |
| --- | --- |
| 最低可接受 | 方案 B 的 OBB 贴合合格率 ≥ 方案 A，且自动通过率 ≥ 30% |
| 目标 | 方案 B/C 的 OBB 贴合合格率 ≥ 70%，自动通过率 ≥ 45% |
| 理想 | 方案 C 的 OBB 贴合合格率 ≥ 85%，自动通过率 ≥ 55%，类别正确率 ≥ 95% |

如果方案 B/C 各项指标不优于方案 A，则暂不推进 v2，继续用 v1 流水线 + 增加人工审核比例。

## 7. 依赖和前置条件

| 依赖项 | 状态 | 负责人 |
| --- | --- | --- |
| GroundingDINO 权重下载和配置 | 待完成 | |
| SAM2 权重（v1 已有 SAM v1，SAM2 需另外下载） | 部分完成 | |
| conda 环境 `openvter-obb-vlm` | 待创建 | |
| v2 主流水线脚本 | 待开发（脚手架已就位） | |
| Qwen/Gemini API Key（方案 C 需要） | 待确认 | |
| 实验数据采样脚本 | 待开发 | |
| 对比分析脚本 | 待开发 | |

## 8. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
| --- | --- | --- | --- |
| GroundingDINO 在 VisDrone 小目标上检测率低 | 中 | 方案 B/C 退化为方案 A | 设 min_crop_size 阈值；小目标直接回退 SAM2 |
| GroundingDINO 安装/环境冲突 | 中 | 阻塞方案 B/C | 独立 conda 环境；Docker 备选 |
| Qwen/Gemini API 不可用或超预算 | 中 | 阻塞方案 C | 方案 C 可降级为只做 offline VLM 审核 |
| SAM2 mask 粘连（多目标） | 高 | OBB 贴合度差 | GroundingDINO refined box 缩小 prompt 范围 |
| tricycle vs awning_tricycle 无法区分 | 中 | 类别准确率低 | 准备 4 类退化版本（合并三轮车类别） |
