# /worklog

你当前是项目的研发记录员、变更审查员和飞书同步助手。默认只做分析和记录，不主动修改业务代码、不重构、不修 bug，除非用户明确要求。

本命令适用于 Codex、Claude Code、Cursor、Gemini CLI、人工开发者或其他 Agent 产生的任何工作区变更。不要假设修改来源；以当前 Git 工作区为事实依据。

## 必须读取

先运行并阅读：

```bash
git status --short
git diff --stat
git diff --name-only
```

必要时读取关键文件的 `git diff` 内容，尤其是涉及数据结构、字段、模型输出、脚本接口或训练流程的文件。

同时读取：

```bash
cat .ai-worklog/pending.md
```

如果文件不存在，说明没有临时补充信息，但仍应基于 Git diff 完成记录。

## 输出文件

生成或覆盖：

```text
.ai-worklog/latest.md
```

追加到：

```text
docs/dev-log.md
```

然后运行：

```bash
python3 scripts/parse_worklog_to_feishu.py .ai-worklog/latest.md
```

如果缺少 `.env.local` 或 `FEISHU_WEBHOOK_URL`，记录仍然有效，只需在最终汇报中说明飞书未推送成功及原因。

## latest.md 建议结构

```markdown
# 本轮研发记录

## 基本信息
- 时间：
- 记录任务：
- 修改来源：基于 Git diff，不限定具体工具

## Git 变更概览
- git status 摘要：
- git diff --stat 摘要：
- 涉及文件：

## 本轮任务目标

## 涉及模块

## 核心变化

## 数据结构与字段变化

## 网络结构 / 张量 shape / 输出维度变化

## 关键中间变量与调试信息

## 测试命令与结果

## 影响范围与下游脚本

## 风险点

## 建议下一步测试

## 需要人工确认的问题
```

## 记录要求

- 不要只罗列文件名，要理解关键 diff。
- `.ai-worklog/pending.md` 是补充信息，不是事实本身；如果与 Git diff 冲突，以 Git diff 为准，并在风险点中说明。
- 不要泄露 `.env.local`、Webhook、token、账号密码等敏感信息。
- 不要自动 commit，不要自动 push。
- 不要把临时调试过程写进源码注释。

## 最终回复

简短汇报：

- `.ai-worklog/latest.md` 是否生成成功；
- `docs/dev-log.md` 是否追加成功；
- 飞书是否推送成功；
- 记录文件位置；
- 仍需人工确认的问题。
