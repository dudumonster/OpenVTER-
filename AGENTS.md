# AGENTS.md

本文件为 Codex、Claude Code、Cursor、Gemini CLI、人工开发者及其他 AI Agent 提供通用协作说明。项目长期背景和规则也会在 `CLAUDE.md` 中维护；如果两者存在冲突，优先遵守用户当前指令，其次遵守更具体的项目文件说明。

## 项目背景

OpenVTER 用于无人机航拍交通视频分析，核心流程包括视频稳定化、图像切分、旋转目标检测、多目标跟踪、轨迹后处理和可视化。近期新增的数据集构建工作集中在 VisDrone HBB 转伪 OBB、车辆/弱势交通参与者 OBB 数据源整理、审核工具和训练数据准备。

## 架构约定

- 主推理入口保留 `video_inference_main.py` 和 `src/train.py`。
- 数据集构建相关脚本集中放在 `dataset_construction/scripts/`。
- 数据集原始文件、下载文件和派生样本默认放在 `dataset_construction/data_sources/` 与 `dataset_construction/derived/`。
- 协作和研发记录文件放在 `.ai-worklog/`、`docs/dev-log.md`、`CLAUDE.md`、`AGENTS.md`。
- 本地私有配置放在 `.env.local`，不要提交 Git。

## 开发规范

- Git diff 是事实依据；不要假设某个改动一定来自某个工具。
- 修改前先理解目标和影响范围，说明准备修改哪些文件以及原因。
- 只修改完成当前任务所必需的文件，不做无关重构。
- 不自动提交 Git commit、不自动 push，除非用户明确要求。
- 不把临时调试过程大量写进源码注释；源码注释只描述稳定逻辑、接口约定和字段含义。

## 测试规范

- 根据改动范围选择最小但有效的测试。
- 修改数据处理脚本时，至少运行对应脚本的帮助命令、干跑或小样本测试。
- 修改飞书/外部服务脚本时，在没有真实配置时测试错误路径，避免泄露密钥。
- 运行过的命令和结果要记录到 `.ai-worklog/pending.md` 或正式工作记录中。

## 记录规范

重要但不适合写入源码注释的信息应记录到 `.ai-worklog/pending.md`，包括：

- 本轮任务目标和修改原因
- 关键变量和中间输出
- 数据结构变化、字段变化、字段顺序变化
- 网络结构变化、张量 shape / 输出维度变化
- 调试过程、测试命令和测试结果
- 风险点、影响范围、待确认问题

整理本轮研发记录时，应基于当前 Git 工作区：

```bash
git status --short
git diff --stat
git diff --name-only
```

必要时读取关键文件 diff，并结合 `.ai-worklog/pending.md` 生成 `.ai-worklog/latest.md`，追加到 `docs/dev-log.md`，再通过飞书同步脚本推送。

## AI Agent 协作方式

- Codex：主要负责代码修改、调试、重构、实验实现，并把关键中间信息写入 `.ai-worklog/pending.md`。
- Claude Code：默认负责变更审查、研发记录整理、项目记忆更新和飞书同步；也可在用户明确要求时参与开发。
- Cursor / Gemini CLI / 人工开发者 / 其他 Agent：产生的修改同样以 Git diff 为准，被纳入统一记录流程。

## 常见命令

```bash
# 查看当前变更
git status --short
git diff --stat
git diff --name-only

# 生成/推送本轮工作记录，通常由 Claude Code 的 /worklog 命令调用
python3 scripts/parse_worklog_to_feishu.py .ai-worklog/latest.md

# VisDrone 伪 OBB 小样本生成
dataset_construction/scripts/run_visdrone_sam_pseudo_obb.sh --splits val --max-images 20

# 源数据集原图与标注抽样
/opt/anaconda3/envs/openvter-obb-sam/bin/python dataset_construction/scripts/export_source_annotation_samples.py --samples-per-dataset 30
```

## 长期维护事项

- `CLAUDE.md`：维护 Claude Code 项目级规则、长期记忆、架构约定和关键背景。
- `AGENTS.md`：维护跨工具通用协作说明。
- `docs/dev-log.md`：长期研发日志。
- `.ai-worklog/pending.md`：临时研发记录，整理后可保留或清理，由团队约定决定。
- `.ai-worklog/latest.md`：最近一次正式工作记录。
