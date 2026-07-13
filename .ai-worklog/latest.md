# OpenVTER 研发日志 — 2026-06-25（第 4 轮）

## 本轮任务目标

Codex 完成了两阶段的 VLM 辅助 OBB 伪标注方案 v2 的设计与实现：

1. **v2 方案设计**：阅读理解 v1 流水线后，设计 v2 方案文档、数据结构（JSONL schema）、质量评分体系、中英文 prompt 模板、实验计划
2. **v2 主流水线实现**：实现核心编排脚本 `vlm_pseudo_obb_v2.py`，含 VLM 后端抽象接口、三种后端实现（Dummy/PureSAM2/GroundingSAM2）、质量评分、审核分类、预览绘制

---

## 涉及模块

| 模块 | 说明 |
| --- | --- |
| `dataset_construction/protocols/` | v2 方案文档和实验计划 |
| `dataset_construction/configs/` | v2 配置文件 |
| `dataset_construction/scripts/` | v2 工具链脚本（5 个新文件） |

---

## 涉及文件

### 第 1 阶段（方案设计）新增

| 文件 | 说明 |
| --- | --- |
| `dataset_construction/protocols/vlm_assisted_visdrone_obb_v2.md` | v2 方案文档（方案对比 A/B/C、架构、类别体系、prompt 模板、审核策略） |
| `dataset_construction/protocols/vlm_v2_experiment_plan.md` | v2 实验计划（4 个 step 的分阶段验证） |
| `dataset_construction/configs/vlm_pseudo_obb_v2.yaml` | v2 配置文件 |
| `dataset_construction/scripts/vlm_pseudo_obb_v2_schema.py` | v2 数据结构、质量评分函数、工具函数 |
| `dataset_construction/scripts/vlm_prompt_templates.py` | VLM/Grounding prompt 模板（中英双语） |
| `dataset_construction/scripts/vlm_pseudo_obb_v2_plan.py` | 环境检查 + dry-run 计划脚本 |
| `dataset_construction/scripts/export_vlm_review_manifest.py` | 审核队列导出脚本 |

### 第 2 阶段（主流水线）新增

| 文件 | 说明 |
| --- | --- |
| `dataset_construction/scripts/vlm_pseudo_obb_v2.py` | v2 核心编排脚本（~500 行），含 `plan` / `generate` 两个子命令 |

### 维持不变

- v1 工具链完整保留（`visdrone_hbb_to_pseudo_obb.py`、`review_pseudo_obb_app.py`、`apply_pseudo_obb_review.py`）
- v1 和 v2 使用独立输出路径，互不影响

---

## 核心代码变化

### v1 → v2 架构差异

```
v1 流水线:  HBB → expand_box → segmenter (SAM/GrabCut) → quality → review
v2 流水线:  HBB → expand_box(0.20) → VLMBackend → Grounding + SAM2 → quality v2 → review

关键变化:
  v1 expand_ratio: 0.10  →  v2 expand_ratio: 0.20 (给 VLM 更多上下文)
  v1 quality_score       →  v2 final_score (几何+语义双维度)
  v1 quality_status      →  v2 review_status (auto_accept/review/reject 三元)
```

### VLMBackend 抽象接口

三种后端实现：

| 后端 | 类名 | 用途 |
| --- | --- | --- |
| Dummy | `DummyVLMBackend` | 无模型测试数据流 |
| PureSAM2 | `PureSAM2Backend` | Baseline（等效 v1 行为） |
| GroundingSAM2 | `GroundingSAM2Backend` | 三阶段：GroundingDINO 检测 → SAM2 分割 |

`GroundingSAM2Backend` 推理流程：
1. GroundingDINO 在 crop 上用 text_prompt 检测，输出 refined box
2. 将 refined box 转为全局坐标，作为 SAM2 box prompt
3. SAM2 生成 mask → 后续 OBB 提取与质量评分

### v2 质量评分体系（双维度）

```
geometry_score = 0.35*area_ratio + 0.25*center_shift + 0.20*fg_ratio + 0.10*aspect + 0.10*boundary
semantic_score = 0.60*class_confidence + 0.40*vlm_box_iou
final_score    = 0.55*geometry + 0.45*semantic
```

审核决策：
- `auto_accept`: final_score >= 0.65 AND geometry >= 0.50 AND semantic >= 0.50
- `review`: 不满足 accept 但 geometry >= 0.30 AND semantic >= 0.30
- `reject`: geometry < 0.30 OR semantic < 0.30

### v2 CLI

```bash
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py plan              # dry-run
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate           # 全量
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate \
  --sample-per-class 100 --splits val --seed 42                             # 小样本
```

---

## 数据结构变化

### VLMPseudoObbRecord（v2 核心记录）

v2 相比 v1 quality.jsonl 的新增字段簇：

**图片级新增**：`annotation_path`, `source_occlusion`, `source_truncation`, `annotation_line_index`

**VLM 级新增**：`vlm_backend`, `text_prompt`, `vlm_box_xyxy`, `vlm_box_confidence`, `vlm_class_name`, `vlm_class_confidence`

**质量级新增**（VLMQualityMetrics）：`boundary_clip_ratio`, `mask_solidity`, `class_confidence`, `vlm_box_confidence`, `vlm_box_iou`, `vlm_class_name`, `vlm_class_agrees_with_hbb`, `geometry_score`, `semantic_score`, `final_score`

**审核级变化**：`review_status` 替代 v1 的 `quality_status`，增加 `queue_reason`, `failure_reasons`

### 关键常量变化

| 参数 | v1 | v2 |
| --- | --- | --- |
| expand_ratio | 0.10 | 0.20 |
| auto_accept 阈值 | 0.55 (单维度) | 0.65 + geometry>=0.50 + semantic>=0.50 (三维度) |
| min_crop_size | 无 | 40px |

---

## 字段变化

v1 `quality.jsonl` vs v2 `quality.jsonl` 不兼容——v2 是 v1 的超集。需要适配的下游工具：
- `review_pseudo_obb_app.py`（Streamlit 审核网页）：需增加 v2 字段展示
- `apply_pseudo_obb_review.py`：需支持 v2 的 `review_status` 字段

---

## 中间变量与调试输出

- VisDrone val split 统计：bicycle 1287, motor 4886, tricycle 1045, awning_tricycle 532，总计 7750 目标
- `vlm_pseudo_obb_v2_plan.py` 运行成功：raw_root 和 split 目录验证通过，数据计数正确
- 预期缺失依赖：GroundingDINO checkpoint、SAM2 checkpoint 未配置；torch/sam2/groundingdino/transformers 在当前环境未安装

---

## 网络结构或输出维度变化

无（v2 使用外部 VLM 模型，不改变 OpenVTER 推理流水线中的网络结构）。

---

## 影响范围

- v1 流水线不受影响，v2 使用独立输出路径
- `apply_pseudo_obb_review.py` 需要适配 v2 的 `review_status` 字段（当前读取 v1 的 `quality_status`）
- `review_pseudo_obb_app.py` 如需展示 v2 的 VLM 相关字段需增加 UI

---

## 可能受影响的下游脚本

| 脚本 | 影响 |
| --- | --- |
| `review_pseudo_obb_app.py` | 需适配 v2 新增字段展示 |
| `apply_pseudo_obb_review.py` | 需支持 `review_status` 替代 `quality_status` |
| `validate_yolo_obb_dataset.py` | 无影响（标签格式不变） |

---

## 已运行测试

| 测试项 | 命令 | 结果 |
| --- | --- | --- |
| 语法检查 ×4 | `python3 -m py_compile` (schema, prompts, plan, export) | 全部通过 |
| 语法检查 | `python3 -m py_compile vlm_pseudo_obb_v2.py` | 通过 |
| plan dry-run | `vlm_pseudo_obb_v2.py plan` | 通过，环境和数据验证正常 |
| CLI help | `vlm_pseudo_obb_v2.py --help` | 正常输出 |
| VisDrone validate | `visdrone_hbb_to_pseudo_obb.py validate` | 通过 |

---

## 测试结果

全部通过（语法检查、plan dry-run）。沙箱环境无 GPU/torch/SAM2，无法运行 `generate` 模式——需在 GPU 服务器上验证。

---

## 风险点

1. **依赖环境**：GroundingDINO + SAM2 + torch + transformers 需要独立 conda 环境（建议 `openvter-obb-vlm`），避免与 `openvter-obb-sam` 冲突
2. **模型权重**：GroundingDINO checkpoint 和 SAM2 checkpoint 待下载和配置路径
3. **API 签名**：`GroundingDINO.predict_with_classes()` 的精确签名需要在真实环境中验证和适配
4. **向下兼容**：v2 的 `quality.jsonl` 与 v1 不兼容，如果后续 Streamlit 审核工具需要同时支持 v1+v2，需要写适配层
5. **VLM 效果未知**：GroundingDINO 在无人机俯视图（小目标、密集场景）上的检测框精度未经验证
6. **Qwen/Gemini 方案**：方案 C 需要 API 预算和网络访问，暂未实施

---

## 建议下一步操作

```bash
# 1. 创建独立 conda 环境
conda create -n openvter-obb-vlm python=3.10 -y
conda activate openvter-obb-vlm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install groundingdino segment-anything pycocotools

# 2. 下载模型权重后跑小样本实验
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate \
  --backend groundingdino_sam2 \
  --sample-per-class 100 --splits val --seed 42

# 3. 查看 v2 质量分布
python3 dataset_construction/scripts/export_vlm_review_manifest.py

# 4. 启动审核界面（需先适配 v2 字段）
streamlit run dataset_construction/scripts/review_pseudo_obb_app.py
```

---

## 需要人工确认的问题

1. GroundingDINO 权重和 SAM2 权重的下载地址和存放路径是什么？
2. 是否需要创建独立 conda 环境 `openvter-obb-vlm`？
3. Qwen/Gemini API（方案 C）是否可用、预算是否允许？
4. GroundingDINO 的 `predict_with_classes()` 方法在 `vlm_pseudo_obb_v2.py` 中的调用签名是否与实际 API 匹配？建议 Codex 在 GPU 服务器上验证
5. 实验计划中方案 A 的 baseline 是用 PureSAM2Backend 还是直接复用 v1 脚本？
