# AI 临时研发记录

这个文件用于记录 Codex/AI 在本项目中进行代码修改、调试、数据处理时产生的关键中间信息。它不是正式文档，优先保证准确、可追溯。

## 记录规则

- 每轮修改前先说明目标、准备修改的文件和原因。
- 业务逻辑、接口约定、稳定字段含义可以写入源码或正式文档。
- 临时调试信息、判断过程、测试输出、风险点统一写在这里。
- 不自动提交 Git commit，除非用户明确要求。

## 当前记录

### 2026-06-15 协作规则落地

**修改目标**

- 建立后续代码修改与调试的临时记录文件，方便追踪数据结构、字段、shape、调试输出、测试结果和待确认问题。

**修改文件**

- `.ai-worklog/pending.md`

**核心变化**

- 新增临时研发记录文件。
- 后续涉及业务代码、配置、脚本、数据处理、模型结构、训练流程的修改，都在本文件追加关键中间信息。

**数据结构变化**

- 无业务数据结构变化。
- 新增记录结构建议如下：
  - 修改目标
  - 修改文件
  - 核心变化
  - 新增或修改的数据结构
  - 字段名称、字段含义、字段顺序变化
  - 网络结构变化
  - 张量 shape / 输出维度变化
  - 关键中间变量
  - 调试输出
  - 临时判断
  - 可能影响下游脚本的地方
  - 运行过的测试命令和测试结果
  - 待确认问题

**测试情况**

- 未运行功能测试；本轮只新增记录文件。

**待确认问题**

- `.ai-worklog/pending.md` 是否需要纳入 Git 版本控制，还是加入 `.gitignore`。当前暂未处理。

### 2026-06-15 项目协作目录与忽略规则

**修改目标**

- 按用户要求建立 Codex/Claude 协作常用目录，并补充本地环境文件与 AI 临时文件的忽略规则。

**修改文件**

- `.gitignore`
- `.claude/commands/.gitkeep`
- `.ai-worklog/pending.md`
- `docs/dev-log.md`

**核心变化**

- 确认/创建目录：`.ai-worklog/`、`docs/`、`scripts/`、`.claude/commands/`。
- 确认/创建文件：`.ai-worklog/pending.md`、`docs/dev-log.md`。
- `.gitignore` 新增：
  - `.env.local`
  - `.ai-worklog/*.tmp`
- 新增 `.claude/commands/.gitkeep`，用于让 Git 能保留空的 Claude 命令目录。

**新增或修改的数据结构**

- 无业务数据结构变化。

**字段名称、字段含义、字段顺序变化**

- 无。

**网络结构变化**

- 无。

**张量 shape / 输出维度变化**

- 无。

**关键中间变量**

- 无。

**调试输出**

- `find .ai-worklog docs scripts .claude -maxdepth 2 -type d -print | sort` 已确认目录存在。

**临时判断**

- `.ai-worklog/pending.md` 按用户说明可以提交，也可以不提交；当前未加入 `.gitignore`。
- `.claude/commands/` 是空目录，Git 默认不跟踪，因此添加 `.gitkeep`。

**可能影响下游脚本的地方**

- `.env.local` 后续不会被 Git 跟踪，适合放本地私有环境变量。
- `.ai-worklog/*.tmp` 后续不会被 Git 跟踪，但 `.ai-worklog/pending.md` 仍可被跟踪。

**运行过的测试命令和测试结果**

- `mkdir -p .ai-worklog docs scripts .claude/commands && touch .ai-worklog/pending.md docs/dev-log.md`
  - 结果：执行成功。
- `find .ai-worklog docs scripts .claude -maxdepth 2 -type d -print | sort`
  - 结果：确认目标目录存在。

**待确认问题**

- 是否需要把 `.ai-worklog/pending.md` 作为团队共享研发记录提交到 Git；当前保持可提交状态。

### 2026-06-15 AI 协作研发记录工作流搭建

**修改目标**

- 搭建适用于 Codex、Claude Code、Cursor、Gemini CLI、人工开发者和其他 AI Agent 的统一研发记录工作流。
- 以 Git diff 作为事实依据，保留 `.ai-worklog/pending.md` 中的临时研发信息，并支持飞书同步。

**修改文件**

- `CLAUDE.md`
- `AGENTS.md`
- `.claude/commands/worklog.md`
- `scripts/parse_worklog_to_feishu.py`
- `.env.example`
- `.ai-worklog/latest.md`
- `docs/dev-log.md`
- `.ai-worklog/pending.md`

**核心变化**

- `CLAUDE.md` 保留原有项目架构说明，在末尾追加 AI 协作研发记录工作流。
- 新增 `AGENTS.md`，提供跨 Codex、Claude Code、Cursor、Gemini CLI、人工开发者的通用协作说明。
- 重写 `.claude/commands/worklog.md`，明确 `/worklog` 需要读取 Git diff、读取 pending、生成 latest、追加 dev-log 并推送飞书。
- 重写 `scripts/parse_worklog_to_feishu.py`，使用 Python 标准库读取 `.env.local` 并推送 Markdown 到飞书机器人。
- `.env.example` 改为只包含 `FEISHU_WEBHOOK_URL="请在这里填写飞书机器人 Webhook"`。
- 新增 `.ai-worklog/latest.md` 占位文件。
- `docs/dev-log.md` 已有长期研发日志内容，本轮保留原内容，不覆盖。
- 检查发现 `docs/dev-log.md` 已存在长期研发日志内容，因此移除了本轮误追加到末尾的重复占位说明，保留原日志结构。

**新增或修改的数据结构**

- 无业务数据结构变化。
- 新增工作记录文件结构：
  - `.ai-worklog/pending.md`：临时补充信息。
  - `.ai-worklog/latest.md`：最近一次正式工作记录。
  - `docs/dev-log.md`：长期追加式研发日志。

**字段名称、字段含义、字段顺序变化**

- 无业务字段变化。
- `.env.local` / `.env.example` 使用字段：
  - `FEISHU_WEBHOOK_URL`：飞书机器人 Webhook 地址。本地私有配置，不提交 Git。

**网络结构变化**

- 无。

**张量 shape / 输出维度变化**

- 无。

**关键中间变量**

- 飞书脚本关键常量：
  - `ENV_FILE = ".env.local"`
  - `ENV_KEY = "FEISHU_WEBHOOK_URL"`
  - `MAX_MESSAGE_CHARS = 18000`

**调试输出**

- 待补充测试后记录。

**临时判断**

- `CLAUDE.md` 已存在且包含重要项目架构说明，因此采用追加方式，不覆盖原内容。
- `.claude/commands/worklog.md` 和 `scripts/parse_worklog_to_feishu.py` 已有雏形，但为满足通用多 Agent 工作流和标准库要求，本轮进行了补强/重写。
- `.gitignore` 已经包含 `.env.local` 与 `.ai-worklog/*.tmp`，本轮不重复添加。
- 飞书推送脚本先校验 Markdown 文件，再读取 `.env.local`，这样缺少工作记录文件时能优先给出文件错误。

**可能影响下游脚本的地方**

- `scripts/parse_worklog_to_feishu.py` 的 CLI 仍保持 `python3 scripts/parse_worklog_to_feishu.py .ai-worklog/latest.md`。
- 脚本现在默认从当前工作目录读取 `.env.local`；如果从其他目录执行，可用 `--project-root` 指定项目根目录。

**运行过的测试命令和测试结果**

- `python3 -m py_compile scripts/parse_worklog_to_feishu.py`
  - 结果：通过，无语法错误。
- `python3 scripts/parse_worklog_to_feishu.py --help`
  - 结果：通过，正常输出命令行帮助。
- `python3 scripts/parse_worklog_to_feishu.py .ai-worklog/latest.md; test $? -eq 1`
  - 结果：通过；在未配置 `.env.local` 时返回错误码 1，并提示复制 `.env.example` 为 `.env.local`、填写 `FEISHU_WEBHOOK_URL`。
- `python3 scripts/parse_worklog_to_feishu.py .ai-worklog/not-exist.md; test $? -eq 1`
  - 结果：通过；Markdown 文件不存在时返回错误码 1，并输出清晰错误。
- `chmod +x scripts/parse_worklog_to_feishu.py`
  - 结果：通过；脚本具备可执行权限。

**待确认问题**

- 飞书 Webhook 需要用户在 `.env.local` 中自行填写。
- 是否每次 `/worklog` 后清空 `.ai-worklog/pending.md`，当前未自动处理，建议由团队约定。

### 2026-06-26 VLM pseudo OBB v2 环境搭建与 PureSAM2 fallback 实验

**任务目标**

- 按用户指令创建独立环境 `openvter-obb-vlm`，安装 v2 伪 OBB 实验依赖。
- 优先尝试方案 B（GroundingDINO + SAM2），如 GroundingDINO checkpoint 不可用则按用户给出的退路跑通 PureSAM2 v2 数据流。

**修改文件**

- `dataset_construction/configs/vlm_pseudo_obb_v2.yaml`
- `dataset_construction/scripts/vlm_pseudo_obb_v2_schema.py`
- `dataset_construction/scripts/export_vlm_review_manifest.py`
- `dataset_construction/protocols/vlm_v2_experiment_plan.md`
- `.ai-worklog/pending.md`

**环境与依赖**

- 新建 conda 环境：`/opt/anaconda3/envs/openvter-obb-vlm`，Python 3.10。
- 安装基础依赖：`torch torchvision opencv-python pyyaml pandas streamlit ultralytics`。
- 安装 SAM2：`pip install sam2`。
- 安装 GroundingDINO：`git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO`，首次 `pip install -e` 因 PEP517 build isolation 看不到 torch 失败，改用 `pip install --no-build-isolation -e /tmp/GroundingDINO` 成功。
- 当前机器为 macOS arm64，`torch.backends.mps.is_available()` 为 `True`，无 CUDA；配置改用 `mps`。

**权重与配置**

- SAM2 checkpoint 下载成功：`checkpoints/sam2/sam2.1_hiera_base_plus.pt`，约 323.6 MB。
- GroundingDINO 配置复制成功：`checkpoints/groundingdino/GroundingDINO_SwinT_OGC.py`。
- GroundingDINO checkpoint 未下载成功：
  - GitHub release 链接长时间 0 字节无响应后中止。
  - Hugging Face 备用链接连续连接超时。
- 因此本轮按用户指令中的快速退路切到 `primary_backend: "pure_sam2"`，输出目录为：
  - `dataset_construction/derived/visdrone_pseudo_obb_v2_expA`
  - `dataset_construction/derived/visdrone_yolo_obb_v2_expA`

**代码修正**

- SAM2 PyPI 包的 `build_sam2()` 通过 Hydra 包内 config name 解析配置，不能使用普通绝对 YAML 路径；配置改为 `configs/sam2.1/sam2.1_hiera_b+.yaml`。
- `vlm_pseudo_obb_v2_schema.py` 修复 JSONL 写出时 `numpy.float32` 等 numpy 类型无法序列化的问题。
- PureSAM2 无 VLM 语义信号时，原评分逻辑 `semantic_score=0` 导致 `auto_accept` 不可能出现；已改为无 VLM 信号时 `semantic_score=1.0` 且 `final_score=geometry_score`，有 VLM 信号时仍保持原几何+语义加权。
- `export_vlm_review_manifest.py` 兼容 v2 字段 `target_class_name` / `source_class_name`，避免 review summary 全部分到 `unknown`。

**测试命令和结果**

- `conda run -n openvter-obb-vlm python dataset_construction/scripts/vlm_pseudo_obb_v2.py plan --config dataset_construction/configs/vlm_pseudo_obb_v2.yaml`
  - 结果：通过；SAM2 checkpoint ok，SAM2/GroundingDINO/transformers/ultralytics import ok；val 目标样本数符合预期。
- smoke test：`python dataset_construction/scripts/vlm_pseudo_obb_v2.py generate --config dataset_construction/configs/vlm_pseudo_obb_v2.yaml --splits val --sample-per-class 1 --seed 42`
  - 初次失败：SAM2 绝对配置路径无法被 Hydra 找到。
  - 第二次失败：JSON 序列化 numpy 类型失败。
  - 修正后通过：4 records，1 auto_accept，3 review，0 reject。
- 正式 PureSAM2 baseline：`python dataset_construction/scripts/vlm_pseudo_obb_v2.py generate --config dataset_construction/configs/vlm_pseudo_obb_v2.yaml --splits val --sample-per-class 100 --seed 42`
  - 结果：400 records，149 auto_accept，249 review，2 reject，review_queue 250。
- `python dataset_construction/scripts/export_vlm_review_manifest.py --pseudo-root dataset_construction/derived/visdrone_pseudo_obb_v2_expA --csv`
  - 结果：导出 `review_manifest.jsonl`、`review_manifest.csv`、`review_summary.json`。

**自动指标**

- 全部：149/400 auto_accept，自动通过率 37.25%。
- bicycle：100 records，21 auto_accept，78 review，1 reject，0 mask_empty，平均 final_score 0.5836。
- motor：100 records，26 auto_accept，73 review，1 reject，0 mask_empty，平均 final_score 0.6201。
- tricycle：100 records，53 auto_accept，47 review，0 reject，0 mask_empty，平均 final_score 0.6948。
- awning_tricycle：100 records，49 auto_accept，51 review，0 reject，0 mask_empty，平均 final_score 0.6865。

**视觉初筛**

- 生成 contact sheet 到 `dataset_construction/derived/visdrone_pseudo_obb_v2_expA/visual_eval/`。
- auto_accept 抽样：awning_tricycle 30/49，bicycle 21/21，motor 26/26，tricycle 30/53。
- review 抽样：每类 30。
- contact sheet 初筛结论：三轮/带篷三轮整体比 bicycle/motor 稳；bicycle/motor 因目标小、密集，存在框过大或包含邻近目标风险。review 队列中仍有可救样本，也有明显框过大/多目标粘连样本。未做逐图精确人工计数。

**待确认问题**

- 需要在网络可访问时补齐 `checkpoints/groundingdino/groundingdino_swint_ogc.pth` 后再运行方案 B。
- 当前实验计划第 5 节已填写 A 方案自动指标；B/C 和精确人工指标仍待后续补齐。

### 2026-06-26 Grounded-SAM-2 方案 B 重试

**任务目标**

- 回答“方案 B 是否有必要”，并在可行时重新测试 GroundingDINO + SAM2。

**新增环境变化**

- 使用 `huggingface_hub.hf_hub_download` 成功下载 GroundingDINO checkpoint：
  - `checkpoints/groundingdino/groundingdino_swint_ogc.pth`
  - 文件大小：693,997,677 bytes。
- B smoke test 首次失败原因：`transformers==5.12.1` 与 GroundingDINO 不兼容，`BertModel` 缺少 `get_head_mask`。
- 已降级依赖：
  - `transformers==4.30.2`
  - `huggingface-hub==0.36.2`
  - `tokenizers==0.13.3`
- GroundingDINO 运行时提示 `Failed to load custom C++ ops. Running on CPU mode Only!`，因此本机 macOS/MPS 环境下 B 明显慢于 A。

**配置变化**

- `dataset_construction/configs/vlm_pseudo_obb_v2.yaml` 恢复为 B 默认：
  - `primary_backend: "grounded_sam2"`
  - `pseudo_root: "dataset_construction/derived/visdrone_pseudo_obb_v2"`
  - `yolo_root: "dataset_construction/derived/visdrone_yolo_obb_v2"`
  - `groundingdino.enabled: true`
  - GroundingDINO/SAM2 checkpoint 路径均已填入。

**测试命令和结果**

- `python dataset_construction/scripts/vlm_pseudo_obb_v2.py plan --config dataset_construction/configs/vlm_pseudo_obb_v2.yaml`
  - 结果：通过；GroundingDINO checkpoint ok，SAM2 checkpoint ok。
- B smoke test：`python dataset_construction/scripts/vlm_pseudo_obb_v2.py generate --config /tmp/openvter_vlm_pseudo_obb_v2_expB.yaml --splits val --sample-per-class 1 --seed 42`
  - 降级 transformers 后通过：4 records，1 auto_accept，3 review，0 reject。
- B 正式小样本：`python dataset_construction/scripts/vlm_pseudo_obb_v2.py generate --config /tmp/openvter_vlm_pseudo_obb_v2_expB_full.yaml --splits val --sample-per-class 100 --seed 42`
  - 结果：400 records，111 auto_accept，289 review，0 reject，review_queue 290。
- B 审核清单：
  - `dataset_construction/derived/visdrone_pseudo_obb_v2/review_manifest.csv`
  - `dataset_construction/derived/visdrone_pseudo_obb_v2/review_summary.json`
- B contact sheets：
  - `dataset_construction/derived/visdrone_pseudo_obb_v2/visual_eval/`

**A/B 自动指标对比**

- A PureSAM2：149/400 auto_accept = 37.25%。
- B GroundingDINO+SAM2：111/400 auto_accept = 27.75%。
- B 有 VLM signal 的样本：376/400；这些样本的 `vlm_class_agrees_with_hbb` 为 376/376。
- B geometry_score 高于 A，但 final_score 被 semantic_score 拉低：
  - awning_tricycle：A final 0.6865 / geom 0.6865；B final 0.6457 / geom 0.7133。
  - bicycle：A final 0.5836 / geom 0.5836；B final 0.5831 / geom 0.6373。
  - motor：A final 0.6201 / geom 0.6201；B final 0.6124 / geom 0.6541。
  - tricycle：A final 0.6948 / geom 0.6948；B final 0.6596 / geom 0.7404。

**临时判断**

- B 值得保留为实验路线，因为几何质量指标确实提高，尤其对小目标可能更保守、更贴合。
- 但按当前阈值和 final_score 加权，B 自动通过率低于 30%，不满足“v2 路线可行”的最低自动通过率条件。
- 下一步不建议直接全量 train 跑 B；建议先逐图审核 A/B contact sheet 或调整 B 的 semantic_score/auto_accept 规则，再决定是否推进。

### 2026-06-26 A/B 可视化审核页面重做

**任务目标**

- 用户反馈原 Streamlit 页面和 contact sheet 看不清标注，需要优化 A/B 效果图和审核界面。

**修改文件**

- `dataset_construction/scripts/review_pseudo_obb_app.py`
- `.ai-worklog/pending.md`

**核心变化**

- 重写 Streamlit 页面为两种模式：
  - `A/B 对比`：默认并排显示 A PureSAM2 与 B Grounded-SAM2。
  - `单方案审核`：用于逐样本审核与保存 accept/edit/reject。
- A/B 对比默认显示放大 ROI，不再只显示整张航拍图。
- 图内只保留标注元素，不再把大段类别/分数文字盖在目标上：
  - 黄色框：VisDrone 原始 HBB。
  - 橙色透明区域：SAM2 mask。
  - 绿色框：最终 OBB。
  - 蓝色框：B 方案中的 GroundingDINO refined box。
  - 红点：OBB 角点。
- 分数、状态、backend 等信息移到网页指标卡和图片标题中。
- 兼容 v1/v2 字段：`image_path/source_image`、`source_hbb_xyxy/hbb_xyxy`、`mask_path/mask_crop`、`target_class_name/class_name`。

**新增效果图**

- 生成 A/B 放大效果图目录：
  - `dataset_construction/derived/visdrone_ab_compare_effects/`
- 每类各 12 个差异明显样本，共 48 个单样本 A/B 并排图。
- 每类 contact sheet：
  - `dataset_construction/derived/visdrone_ab_compare_effects/bicycle_contact.jpg`
  - `dataset_construction/derived/visdrone_ab_compare_effects/motor_contact.jpg`
  - `dataset_construction/derived/visdrone_ab_compare_effects/tricycle_contact.jpg`
  - `dataset_construction/derived/visdrone_ab_compare_effects/awning_tricycle_contact.jpg`

**测试情况**

- `/opt/anaconda3/envs/openvter-obb-vlm/bin/python -m py_compile dataset_construction/scripts/review_pseudo_obb_app.py`
  - 结果：通过。
- `curl http://localhost:8501`
  - 结果：HTTP 200，Streamlit 服务仍可访问。
- 抽查 `bicycle_contact.jpg`，确认放大后标注可见，图内文字不再遮挡目标。

### 2026-06-25 VLM 辅助弱势交通参与者 OBB 伪标注方案 v2 设计

**修改目标**

设计 "VLM 辅助弱势交通参与者 OBB 伪标注数据集构建方案" 的非视觉部分：
- 阅读理解 v1 流水线现有代码和文档
- 设计 v2 方案文档（protocol）
- 设计 v2 数据结构和 JSONL schema（含质量评分配置）
- 设计中英文 prompt 模板
- 设计实验计划
- 新增轻量级配置和脚本脚手架
- 所有修改不涉及真实图像可视化、mask 质量判断、OBB 贴合度主观判断（留给 Codex 后续执行）

**修改文件**

新增文件：
- `dataset_construction/protocols/vlm_assisted_visdrone_obb_v2.md` — v2 方案文档
- `dataset_construction/protocols/vlm_v2_experiment_plan.md` — 实验计划
- `dataset_construction/configs/vlm_pseudo_obb_v2.yaml` — v2 配置文件
- `dataset_construction/scripts/vlm_pseudo_obb_v2_schema.py` — v2 数据结构、质量评分和共享工具函数
- `dataset_construction/scripts/vlm_prompt_templates.py` — VLM/Grounding prompt 模板
- `dataset_construction/scripts/vlm_pseudo_obb_v2_plan.py` — 环境检查 + dry-run 计划脚本
- `dataset_construction/scripts/export_vlm_review_manifest.py` — 审核队列导出脚本
- `.ai-worklog/pending.md` — 本轮记录追加

未修改现有文件（v1 流水线完全保留，v2 使用独立路径）。

**核心代码变化**

v2 与 v1 的关系：
- v1 输出路径：`visdrone_pseudo_obb_v1` / `visdrone_yolo_obb_v1`
- v2 输出路径：`visdrone_pseudo_obb_v2` / `visdrone_yolo_obb_v2`
- v2 复用 v1 的部分：VisDrone 数据解析、expand_box、select_component、obb_from_mask、postprocess_mask、order_points_clockwise、normalize_points
- v2 新增的部分：VLM backend abstraction layer（预留接口，未实现具体模型调用）、v2 quality scoring（geometry + semantics）、prompt management、review priority classification

**新增或修改的数据结构**

`VLMPseudoObbRecord` (dataclass) — v2 完整记录，序列化到 `quality.jsonl`：

```
sample_id: str                     # "split__stem__line_index"
source_dataset: str                # "VisDrone2019-DET"
split: str                         # train/val/test-dev
image_path: str
annotation_path: str
image_width/height: int
source_hbb_xywh: [x,y,w,h]        # VisDrone HBB
source_hbb_xyxy: [x1,y1,x2,y2]
source_class_id: int               # VisDrone class_id
source_class_name: str
source_occlusion/truncation: int   # 0/1/2
annotation_line_index: int
target_class_id: int               # OBB training class_id (1-4)
target_class_name: str
crop_box_xyxy: [x1,y1,x2,y2]      # expanded crop global coords
crop_scale: float                  # resize scale for VLM
vlm_backend: str                   # grounded_sam2/florence2_sam2/qwen/gemini/...
text_prompt: str
vlm_box_xyxy: [x1,y1,x2,y2]       # VLM refined box
vlm_box_confidence: float
vlm_class_name: str                # VLM-predicted class
vlm_class_confidence: float
mask_path: str                     # relative path to mask PNG
obb_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
yolo_obb: [class_id, x1_norm, y1_norm, ...]
quality: VLMQualityMetrics
review_status: str                 # auto_accept/review/reject
queue_reason: str
failure_reasons: [str]
preview_path: str
created_at: str
```

`VLMQualityMetrics` (dataclass) — v2 质量指标：

```
# Geometry (v1 inherited)
hbb_area, mask_area, obb_area: float
area_ratio, center_shift, foreground_ratio, aspect_ratio: float

# v2 additions
boundary_clip_ratio: float         # OBB clipped fraction
mask_solidity: float               # mask_area / convex_hull_area

# VLM / semantic metrics
class_confidence: float            # VLM class confidence
vlm_box_confidence: float          # grounding box confidence
vlm_box_iou: float                # IoU between VLM box and SAM2 box
vlm_class_name: str               # VLM-predicted class
vlm_class_agrees_with_hbb: bool

# Composite scores
geometry_score: float              # 0.35*area + 0.25*center + 0.20*fg + 0.10*aspect + 0.10*boundary
semantic_score: float              # 0.60*class_conf + 0.40*vlm_box_iou
final_score: float                 # 0.55*geometry + 0.45*semantic

# Flags
flags: [str]
```

**字段名称、字段含义、字段顺序变化**

v2 相比 v1 quality.jsonl 的新增字段：
- 图片级：`annotation_path`, `source_occlusion`, `source_truncation`, `annotation_line_index`
- VLM 级：`vlm_backend`, `text_prompt`, `vlm_box_xyxy`, `vlm_box_confidence`, `vlm_class_name`, `vlm_class_confidence`
- 质量级（quality 内）：`boundary_clip_ratio`, `mask_solidity`, `class_confidence`, `vlm_box_confidence`, `vlm_box_iou`, `vlm_class_name`, `vlm_class_agrees_with_hbb`, `geometry_score`, `semantic_score`, `final_score`
- 审核级：`review_status`（替代 v1 的 quality_status）, `queue_reason`, `failure_reasons`

v2 相比 v1 的字段变化：
- `v1: expand_ratio` 从 0.10 增加到 v2: 0.20（给 VLM 更多上下文）
- `v1: quality.quality_score` → `v2: quality.final_score`（名称变化）
- `v1: quality.quality_status` → `v2: review_status`（名称变化，含义更精确）
- v2 新增 `min_crop_size=40`（小于此尺寸的裁剪区域跳过 VLM）
- 质量阈值从 v1 的 `auto_accept >= 0.55` 改为 v2 的 `auto_accept: final_score >= 0.65 AND geometry_score >= 0.50 AND semantic_score >= 0.50`

**网络结构变化**

无。

**张量 shape / 输出维度变化**

无。

**关键中间变量**

v2 方案核心变量：
- `VLM_BACKENDS`: tuple of supported backend names
- `VISDRONE_TO_GLOBAL`: {visdrone_class_id: (global_class_id, class_name)} — 与 v1 相同
- `ASPECT_PRIORS`: 类别长宽比先验 — 与 v1 相同
- `PRIORITY_ORDER`: 审核队列优先级排序（class_conflict > intra_class_confusion > occlusion_truncation > geometry_low_semantic_high > random_sample）

VisDrone val 目标统计数据（run 验证）：
- bicycle: 1287
- motor: 4886
- tricycle: 1045
- awning_tricycle: 532
- 总计: 7750 个目标

**调试输出**

- `python3 -m py_compile` 四个新脚本全部通过（无语法错误）
- `python3 dataset_construction/scripts/vlm_pseudo_obb_v2_plan.py` 运行成功：
  - raw_root 和所有 split 目录验证通过
  - 数据计数正确
  - 预期问题：GroundingDINO checkpoint、SAM2 checkpoint 未配置
  - 预期问题：torch/sam2/groundingdino/transformers/ultralytics 在当前 conda 环境未安装（非 openvter-obb-sam 环境）

**临时判断**

- v1 脚本 `visdrone_hbb_to_pseudo_obb.py` 的架构设计良好，v2 可以复用其 Segmenter 抽象模式来设计 VLM 后端接口
- 不建议在 v2 中完全重写 v1 的生成逻辑，而是新增 VLM 前端 + 复用 v1 的后端（分割→OBB→质量评分→审核）
- v2 质量评分中的 `mask_solidity`（mask 面积/凸包面积）是 v1 没有的指标，可以帮助检测碎片化 mask
- prompt 模板中包含了区分规则（如 tricycle vs awning_tricycle），但 VLM 实际能否在俯视小图中区分这些类别，需要 Codex 视觉验证

**可能影响下游脚本的地方**

- v2 新增的 `quality.jsonl` 格式与 v1 不完全兼容（v2 字段更多），但 v1 的 `review_pseudo_obb_app.py` 默认读取 v1 路径，不受影响
- 如果后续要让 Streamlit 审核网页同时支持 v2，需要在 `review_pseudo_obb_app.py` 中增加 v2 字段的展示逻辑
- `apply_pseudo_obb_review.py` 读取 `quality.quality_status` 字段判断 auto_accept；v2 使用 `review_status` 字段，需要适配或增加兼容逻辑
- 新增的 `vlm_pseudo_obb_v2_schema.py` 中的 `score_v2_quality()` 和 `classify_review_status()` 是 v2 流水线的核心函数，后续 v2 主脚本应调用这些函数而非 v1 的 `score_quality()`

**运行过的测试命令和测试结果**

1. `python3 -m py_compile dataset_construction/scripts/vlm_prompt_templates.py` → 通过
2. `python3 -m py_compile dataset_construction/scripts/vlm_pseudo_obb_v2_schema.py` → 通过
3. `python3 -m py_compile dataset_construction/scripts/vlm_pseudo_obb_v2_plan.py` → 通过
4. `python3 -m py_compile dataset_construction/scripts/export_vlm_review_manifest.py` → 通过
5. `python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py validate` → 通过，确认了 val split 中 4 类目标数量
6. `python3 dataset_construction/scripts/vlm_pseudo_obb_v2_plan.py` → 通过，数据路径验证成功，正确识别缺失的模型权重和依赖

**待确认问题**

1. GroundingDINO 权重放置路径和配置文件路径待确认。
2. SAM2 权重是否需要单独下载（v1 使用的是 SAM v1 vit_b）。
3. 是否创建独立 conda 环境 `openvter-obb-vlm`，避免与 v1 的 `openvter-obb-sam` 冲突。
4. Qwen/Gemini API 是否可用、预算是否允许（方案 C 需要）。
5. v2 主流水线脚本的 Segmenter 抽象如何扩展以支持 GroundingDINO + SAM2 双阶段推理。
6. Streamlit 审核网页是否需要为 v2 新增字段做适配。
7. 实验计划中方案 A 的 baseline 是直接复用 v1 脚本还是需要统一用 v2 脚本关掉 VLM 后端。

**需要 Codex 后续视觉验证的任务**

已在新方案文档 §14 中完整列出，主要包括：
- 14.1 分割质量对比（v1 vs v2）
- 14.2 VLM 类别判断准确率
- 14.3 OBB 贴合度人工评定
- 14.4 Prompt 有效性测试
- 14.5 后端对比（GroundingDINO vs Florence-2）

**建议下一步开发任务**

1. 搭建 conda 环境并安装 GroundingDINO + SAM2 依赖。
2. 实现 v2 主流水线脚本（可参考 v1 架构，新增 Grounding 前端 + 复用 SAM2 后端）。
3. 运行第一批小样本实验（实验计划 Step 0-2，每类 100 样本）。
4. Codex 视觉评估后，根据结果调整 prompt 模板、质量阈值和审核队列策略。
5. 如需方案 C，接入 Qwen/Gemini API 做语义审核。

### 2026-06-25 v2 主流水线脚本实现

**修改目标**

实现 v2 核心编排脚本 `vlm_pseudo_obb_v2.py`，提供 `plan` 和 `generate` 两个子命令。
设计要点：
- v2 重写编排层，不复用 v1 的 `Segmenter` 类——v2 用 `VLMBackend` 抽象接口统一 Grounding+SAM2 双阶段推理
- 实现三种后端：`DummyVLMBackend`（无模型测试用）、`PureSAM2Backend`（baseline，等效 v1 SAM2）、`GroundingSAM2Backend`（GroundingDINO + SAM2）
- 实现 `generate_v2()` 编排函数，含采样（`--sample-per-class N`）、v2 质量评分、v2 预览绘制（含 VLM box 蓝色虚线）
- 生成 v2 manifest.json，包含 auto_accept/review/reject 统计

**修改文件**

新增：
- `dataset_construction/scripts/vlm_pseudo_obb_v2.py` — v2 主流水线脚本（约 500 行）

**核心代码变化**

- `VLMBackend` 抽象基类：`refine_and_segment(image, hbb_xyxy, expanded_xyxy, class_name) -> dict`
  - 返回字典包含 `mask`, `vlm_box_xyxy`, `vlm_box_confidence`, `vlm_class_name`, `vlm_class_confidence`, `text_prompt`
- `PureSAM2Backend(name="pure_sam2")`：直接使用 expanded_xyxy 作为 SAM2 box prompt，不含 VLM。等效 v1 行为，作为实验 baseline。
- `GroundingSAM2Backend(name="groundingdino_sam2")`：三阶段推理
  1. GroundingDINO 在 crop 上用 text_prompt 检测，获得 refined box
  2. 将 refined box（或 fallback expanded HBB）转为全局坐标，作为 SAM2 box prompt
  3. SAM2 生成 mask
- `DummyVLMBackend(name="dummy")`：返回矩形 mask（从 HBB 生成），无模型依赖，用于测试数据流

`generate_v2(config, args)` 主函数特点：
- 支持 `--sample-per-class N`：先扫描全部 sample_id，再按类别均匀随机采样
- 质量评分使用 `score_v2_quality()`（几何+语义双维度）
- review 决策使用 `classify_review_status()`（auto_accept/review/reject 三元分类）
- 预览图 `draw_preview_v2()` 增加 VLM 蓝色虚线框（GroundingDINO 输出）展示
- 输出 `quality.jsonl`、`review_queue.jsonl`、YOLO-OBB 标签、`manifest.json`

CLI：
```bash
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py plan          # dry-run
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate       # 全量运行
python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate \
  --sample-per-class 100 --splits val --seed 42                         # 小样本实验
```

**数据结构变化**

无新增数据结构（复用 `vlm_pseudo_obb_v2_schema.py` 中的 `VLMPseudoObbRecord` 和 `VLMQualityMetrics`）。

manifest.json 新增字段：
```json
{
  "version": "v2",
  "backend": "groundingdino_sam2",
  "config": { ... },
  "totals": {
    "records": N,
    "auto_accept": N,
    "review": N,
    "reject": N,
    "review_queue": N
  },
  "counters": { ... }
}
```

**字段名称、字段含义、字段顺序变化**

无（manifest 新增字段，不影响 quality.jsonl schema）。

**调试输出**

- `python3 -m py_compile dataset_construction/scripts/vlm_pseudo_obb_v2.py` → 通过
- `python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py plan` → 通过，正确识别环境和数据状态
- `python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py --help` → 通过，正常输出 CLI help

**测试情况**

沙箱环境无 GPU/torch/SAM2，无法运行 `generate` 模式。`plan` 模式和 `py_compile` 通过。

**待确认问题**

- GroundingDINO API (`groundingdino.util.inference.Model`) 的 `predict_with_classes()` 方法具体签名需在真实环境中验证和适配
- SAM2 predictor 的 `set_image()` 在每帧只调用一次的缓存策略假设大图（不切 tile），如果后续改为 tile 模式需要调整
- v2 预览图 `draw_preview_v2()` 新增 VLM 蓝色虚线框，但 Streamlit 审核网页 (`review_pseudo_obb_app.py`) 尚未适配 v2 字段，需要后续增加 v2 字段展示逻辑
- `GroundingSAM2Backend.is_available()` 只检查 import，不检查权重文件是否存在

### 2026-06-26 本地与 GitHub main 同步

**任务目标**

- 按用户要求以 GitHub `origin/main` 为基准，同步远端最新提交，同时保留本地新增的数据集构建、协作记录和飞书同步相关改动。
- 将综合后的最新状态提交并推送到 GitHub，方便其他平台拉取。

**同步过程**

- 当前分支：`main`。
- 同步前状态：本地 `main` 落后 `origin/main` 2 个提交，工作区有本地未提交修改和未跟踪文件。
- 创建保护分支：`backup/local-before-sync-20260626-111410`。
- 执行 `git stash push -u -m "local work before syncing with origin/main"` 保护本地改动。
- 遇到 `.git/index.lock` 残留；确认无写入型 Git 进程后删除锁文件并重试 stash。
- 执行 `git pull --ff-only origin main`，将远端 2 个提交 fast-forward 到本地。
- 执行 `git stash pop`，本地改动叠回最新 `main`，无冲突。

**关键判断**

- `.env.local` 为本地私有配置，继续由 `.gitignore` 忽略，不提交。
- `dataset_construction/data_sources/*/raw/*`、`downloads/*`、`derived/*`、`logs/*` 为数据集原始文件、下载文件、派生数据和运行日志，由 `dataset_construction/.gitignore` 忽略，不提交。
- `tmp/` 为临时预览输出，本轮补充到根 `.gitignore`，不提交。

**已运行验证**

- `python3 -m py_compile dataset_construction/scripts/*.py scripts/parse_worklog_to_feishu.py scripts/feishu_mcp.py`
  - 结果：通过。
- `python3 scripts/parse_worklog_to_feishu.py --help`
  - 结果：通过。
- `python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py --help`
  - 结果：系统默认 Python 缺少 `cv2`，失败。
- `/opt/anaconda3/envs/openvter-obb-sam/bin/python dataset_construction/scripts/vlm_pseudo_obb_v2.py --help`
  - 结果：通过。
- `node tests/test_dimension_analysis.js`
  - 结果：通过，输出 `dimension analysis tests passed`。
- `python3 -m pytest tests/test_calibration_and_dimension_filter.py`
  - 结果：失败，系统默认 Python 缺少 `pytest`。
- `/opt/anaconda3/envs/openvter-obb-sam/bin/python -m pytest tests/test_calibration_and_dimension_filter.py`
  - 结果：失败，`openvter-obb-sam` 环境缺少 `pytest`。

**风险点**

- Python 端远端新增测试尚未实际执行，原因是当前可用 Python 环境未安装 `pytest`。
- 本地仍保留大量被忽略的数据集文件和派生结果；它们不会随本轮提交上传到 GitHub。

### 2026-07-14 容器内双链路轨迹拖尾视频

**任务目标与修改原因**

- 为 `ban_xian_shan_001` 增加两条视频后处理链路：处理后 `moving_filtered` OBB 回映原视频，以及原始 `tracking_output_stab_det_*` 视频拖尾。
- 每条链路支持 10 秒有限拖尾和永久拖尾；原视频版不显示 ID，tracking 版不新增 ID。
- 所有服务器命令使用容器内绝对路径，并把输出写入 visualization 持久化目录。

**核心实现**

- 扩展 `scripts/overlay_processed_obb_on_video.py`，新增 `--video-source`、`--trail-mode`、拖尾时长/宽度、类别图例、OBB 开关和持久化 artifact root。
- 有限拖尾按帧龄淡出，目标离场后继续保留至 10 秒生命周期结束；永久拖尾使用稳像坐标持久画布增量累计。
- 原视频模式将稳像拖尾层和 OBB 使用当前帧逆稳像矩阵映射回原图；tracking 模式使用 PKL 的像素 OBB 中心和 `output_frame -> source_frame` 映射。
- 新增 `scripts/run_ban_xian_shan_001_trails.sh`，提供容器路径、依赖、可写性、8 GiB 空间预检，并分别执行 preview/full 四种输出。
- 根据原始 PKL 统计，30 秒最密集预览窗口为 2079-2977 帧，共 899 帧，平均约 53.13 个目标/帧。

**输出数据与参数**

- 输入视频：3840x2160，29.97 FPS，9024 帧。
- 有限拖尾：10 秒，按结果帧率换算为 300 帧；线宽 4；最大连接缺口 30 帧。
- tracking 原始 PKL：9024 个完整帧映射，1146 条 raw track，428042 行观测。
- moving_filtered：181 条处理后 track，111638 行轨迹。
- 输出报告包含输入路径、帧映射数、轨迹/类别统计、写入帧数、帧率、分辨率和缺失映射数。

**已运行验证**

- `bash -n scripts/run_ban_xian_shan_001_trails.sh`：通过。
- `python -m py_compile scripts/overlay_processed_obb_on_video.py`：通过。
- `python -m pytest tests/test_trajectory_video_trails.py -q`：5 passed。
- `python -m pytest tests/test_trajectory_video_trails.py tests/test_calibration_and_dimension_filter.py -q`：最终 13 passed；仅有本机 SciPy/NumPy 版本提示，不影响测试结果。
- tracking finite 真实 4K 冒烟：12/12 帧，29.97 FPS，3840x2160，缺失映射 0。
- tracking permanent 真实 4K 冒烟：3/3 帧，29.97 FPS，3840x2160，缺失映射 0。
- moving_filtered 重建 OBB/finite 真实 4K 冒烟：5/5 帧，14 个框/帧，缺失映射 0；本地使用稳像视频作底图，仅验证处理链路，真正原视频逆稳像视觉效果留到服务器容器预览确认。
- `git diff --check`：通过，仅有 Windows 工作区 LF/CRLF 提示。

**风险与待确认**

- 本地没有服务器原始 MP4，原视频逆稳像对齐必须先在容器执行 preview 并人工检查，确认后再执行 full。
- 4 个完整 4K 视频预计占用约 4-6 GiB；运行脚本按至少 8 GiB 可用空间预检。
- tracking 输入视频中的 ID 已在推理阶段烧入，后处理只保证不新增 ID，无法直接移除已有文字。

### 2026-07-14 轨迹预览 MP4 完整性与 17 秒拖尾修正

- 有限拖尾时长由 10 秒调整为 17 秒；29.97 FPS 下约为 509 帧。
- 输出视频先写入同目录隐藏文件 `.<输出名>.part.mp4`，关闭编码器并用 OpenCV 校验可读性和帧数后，再原子替换最终 `.mp4`。
- 渲染期间每 100 帧输出一次进度；如果渲染中断或校验失败，只保留 `.part.mp4`，不会让未封装完成的文件冒充最终视频。
- 17 秒有限拖尾输出名统一使用 `trail_17s`；永久拖尾输出名保持不变。
- 无法打开的旧永久预览最可能是在旧版直接写最终文件时被中断，导致 MP4 索引未完成；旧损坏文件需要重新渲染，无法通过改参数原地修复。
- `python -m pytest tests/test_trajectory_video_trails.py tests/test_calibration_and_dimension_filter.py -q`：15 passed；仅有本机 SciPy/NumPy 版本提示。
- 真实 4K、永久拖尾、17 秒默认参数的 3 帧原子发布冒烟测试通过：3840×2160、29.97 FPS、3/3 帧可解码；成功后 `.part.mp4` 不残留，报告中的有限拖尾换算值为 509 帧。
