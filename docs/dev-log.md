# OpenVTER 长期研发日志

## 2026-06-15 — AI 协作体系 + 数据集构建工作区 + 方向规划文档

### 协作基础设施

- 建立 `.ai-worklog/` 目录作为 AI 临时研发记录存放处
  - `pending.md`：Codex/Claude 修改过程中的中间变量、输出维度、网络结构、字段变化、调试信息和人工备注
  - `latest.md`：每轮任务的结构化日志，适合推送飞书
  - `.ai-worklog/*.tmp` 加入 `.gitignore`，`pending.md` 保持可提交
- 创建 `.claude/commands/` 目录并添加 `.gitkeep`，用于 Claude 自定义命令
- `.gitignore` 新增规则：`.env.local`（本地私有环境变量）、`.ai-worklog/*.tmp`

### 研究方向规划

完成三份方向规划文档：

1. **`docs/research_direction_model_plan.md`** — 研究主线与模型训练路线
   - 主线定位：面向中国复杂道路/环岛/人车混行场景，构建多类别交通参与者的航拍轨迹提取框架
   - 模型路线推荐：MMRotate/RTMDet-R > YOLO11-OBB > 继续训练 YOLOX-R
   - 训练策略：八参数 OBB 角点标注、切片微调+切片推理、全局 rotated NMS
   - 推荐实验矩阵：检测模型对比（OBB/HBB）、类别泛化（3类→7类）、轨迹层指标（IDF1, MOTA, 轨迹完整率）
   - 论文创新点组织为三条：(1) 复杂中国道路多类别航拍轨迹数据集 (2) 多类别 OBB 感知训练策略 (3) 检测→轨迹→交互闭环评价

2. **`docs/additional_dataset_papers_parallel_analysis.md`** — 新增数据集论文并列分析
   - 覆盖 7 个数据集：KITTI, Waymo Open, highD, inD, Stanford Drone, SIND, AUTOMATUM DATA
   - 车端感知 vs 航拍轨迹的二分框架
   - SIND（天津信号交叉口）被识别为最接近的对比工作
   - 建议文献组织逻辑：由远到近（车端→高速航拍→城市交叉口→多类别 VRU）

3. **`zotero_classification_report.md`** — Zotero 论文库分类
   - 全量 33 篇论文按主题分为 11 个类别
   - 主要类别：交通数据集与基准(10)、目标检测/小目标/旋转框(8)
   - 标记优先处理项（含置信度评级和分类建议）

### 数据集构建工作区

- 搭建 `dataset_construction/` 完整工作区结构：
  - 数据源清单、转换协议、类别映射 schema、处理脚本、审核工具
- 实现 VisDrone HBB → 伪 OBB 工具链 v1：
  - 核心脚本：`visdrone_hbb_to_pseudo_obb.py`（validate / generate 两个子命令）
  - 支持 SAM ViT-B box-prompt 分割和 OpenCV GrabCut fallback
  - 质量评估：输出 `quality.jsonl`（全量评分）+ `review_queue.jsonl`（低质样本队列）
  - 审核界面：Streamlit app (`review_pseudo_obb_app.py`)
  - 审核应用：`apply_pseudo_obb_review.py`
  - 标签校验：`validate_yolo_obb_dataset.py`
- 5 类 OBB 训练策略：`motor_vehicle`（粗类+车长细分）、`bicycle`、`motor`、`tricycle`、`awning_tricycle`
- 独立 conda 环境：`openvter-obb-sam`（本机 MPS）和 `obb-sam`（服务器 CUDA）
- SAM v1 权重：`checkpoints/sam/sam_vit_b_01ec64.pth`（需单独下载，已加入 .gitignore）
- 本机 MPS 验证通过（5 张 val 图约 9 秒）

### 依赖与环境

- `requirements.txt` 新增 SAM/SAM2 可选注释，帮助团队了解伪 OBB 加速器安装方式

### 修改文件清单

| 文件 | 操作 |
| --- | --- |
| `.ai-worklog/pending.md` | 新建 |
| `.ai-worklog/latest.md` | 新建 |
| `.claude/commands/.gitkeep` | 新建 |
| `docs/dev-log.md` | 新建 |
| `docs/research_direction_model_plan.md` | 新建 |
| `docs/additional_dataset_papers_parallel_analysis.md` | 新建 |
| `zotero_classification_report.md` | 新建 |
| `.gitignore` | 修改（+2 条规则） |
| `requirements.txt` | 修改（+5 行注释） |

### 风险与待确认

- 伪 OBB 对密集小目标的贴合度需要审核确认
- 5 类 coarse-to-fine（粗类+几何细分）策略需要实验验证
- SAM2 与现有 PyTorch 环境的兼容性需要独立 conda 环境隔离
- 邮件询问：`.ai-worklog/` 和 `docs/dev-log.md` 是否应提交 Git（当前保持可提交）

---

## 后续记录模板

每条新记录应包含：日期、目标、涉及模块、核心变化、数据结构变化、风险点、测试结果、待确认问题。

详细内容参见每轮的 `.ai-worklog/latest.md`。

---

## 2026-06-15（第 2 轮）— AI 协作研发记录工作流搭建

### 任务目标

搭建跨 Codex、Claude Code、Cursor、Gemini CLI、人工开发者的统一研发记录工作流。

### 核心变更

- `CLAUDE.md`：末尾追加 AI 协作研发记录工作流章节（+53 行），明确 Claude Code 是默认记录审查员、Git diff 是事实依据、`/worklog` 7 步流程
- `AGENTS.md`：新建跨工具通用协作说明，覆盖项目背景、架构约定、开发/测试/记录规范、Agent 分工
- `.claude/commands/worklog.md`：重写为 8 步标准流程
- `scripts/parse_worklog_to_feishu.py`：纯标准库重写，支持 `--project-root`、自动截断、中文错误提示
- `.env.example`：精简为单行模板

### 测试结果

- 语法检查通过
- 缺少 `.env.local` 错误路径：正确报错退出码 1
- Markdown 缺失错误路径：正确报错退出码 1
- 脚本具备可执行权限

### 风险点

- `.env.local` 为空，飞书推送不可用
- 多 Agent 同时写 `pending.md` 可能冲突
- `.env.example` 格式变更，已有 `.env.local` 需核对

### 待确认

- 飞书 Webhook URL 需人工填入
- 是否每次 `/worklog` 后清空 `pending.md`？

---

## 2026-06-15（第 3 轮）— 源数据集标注抽样审查

### 任务目标

从 VisDrone、VSAI、UAV-OBB 三个数据集各抽取 30 张样本，生成原图+标注可视化对比页面，审查各数据源的标注格式、类别分布和质量。

### 核心变更

- 新增 `export_source_annotation_samples.py`：619 行只读抽样脚本，支持 multi-dataset、随机抽样、接触印相、HTML 总览
- 输出 90 张样本（每数据集 30 张）、contact sheet、对比图对、README 统计
- 三格式标注对比：HBB（VisDrone）vs polygon（VSAI）vs YOLO-OBB 四点（UAV-OBB）

### 类别统计

- VisDrone：12 类（含 people/tricycle/awning-tricycle 等 VRU 细类）
- VSAI：2 类粗分类（small/large vehicle），非机动车缺失
- UAV-OBB：6 类（car/bike/bus/taxi/truck/other），偏重车辆分类

### 运行环境

conda `openvter-obb-sam`，Python 3.10

### 待确认

- VisDrone people/pedestrian 区分标准
- UAV-OBB bike 是否涵盖电动车/摩托车
- VSAI 仅用于 pretrain 还是也可用于 fine-tune
- DOTA/DroneVehicle 数据何时接入
- `dataset_construction/derived/` 大量图片是否需要 gitignore

---

## 2026-06-25（第 4 轮）— VLM 辅助 OBB 伪标注方案 v2 设计与实现

### 任务目标

设计并实现 VLM（视觉语言模型）辅助的弱势交通参与者 OBB 伪标注方案 v2，在 v1 纯几何 pipeline 基础上引入 GroundingDINO + SAM2 语义信息，提升小目标和密集场景的标注质量。

### 核心变更（两阶段）

**第 1 阶段 — 方案设计：**
- 新增 2 份方案文档（v2 方案 + 实验计划）
- 新增 v2 数据结构 `VLMPseudoObbRecord` + `VLMQualityMetrics`（双维度评分：几何 + 语义）
- 新增中英文 prompt 模板
- 新增 v2 配置文件和 dry-run 计划脚本

**第 2 阶段 — 主流水线实现：**
- 新增 `vlm_pseudo_obb_v2.py`（~500 行），含 `plan` / `generate` 两个子命令
- 实现 `VLMBackend` 抽象接口，三种后端：Dummy/PureSAM2/GroundingSAM2
- v2 质量评分为双维度：`geometry_score`(55%) + `semantic_score`(45%)
- 审核分类三元：auto_accept / review / reject
- v2 预览图增加 VLM 蓝色虚线框展示

### 涉及文件

新增 8 个文件（全部在 `dataset_construction/` 下），v1 工具链完整保留不变。

### 数据结构变化

v2 `quality.jsonl` 为 v1 的超集，新增 15+ 字段（VLM 级、语义质量级、审核级）。v1/v2 不兼容。

### 测试结果

全部语法检查通过，plan dry-run 通过。generate 模式需 GPU 服务器验证。

### 待确认

- GroundingDINO + SAM2 权重路径和 conda 环境
- v2 质量阈值（0.65/0.50/0.50）是否合理
- Qwen/Gemini API（方案 C）是否可用
- 下游审核工具（Streamlit）需适配 v2 字段
