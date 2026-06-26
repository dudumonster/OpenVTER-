#!/usr/bin/env python3
"""飞书工作记录同步脚本。

用法:
    python3 scripts/parse_worklog_to_feishu.py .ai-worklog/latest.md

流程：
1. 通过 feishu_mcp.py (TAT 模式) 将 Markdown 内容创建或覆写为一篇飞书文档
2. 通过群聊机器人 Webhook 发送文档链接通知

环境变量（.env.local）：
    FEISHU_APP_ID          飞书自建应用 App ID
    FEISHU_APP_SECRET      飞书自建应用 App Secret
    FEISHU_FOLDER_TOKEN    文档存放目录 token（可选，不填则放根目录）
    FEISHU_WEBHOOK_URL     群聊机器人 Webhook（用于发通知）
    FEISHU_DOC_ID          已创建的飞书文档 ID（首次创建后写入，后续覆写）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ENV_FILE = ".env.local"
FEISHU_MCP = Path(__file__).resolve().parent / "feishu_mcp.py"
KEYWORD_PREFIX = "AI研发记录\n\n"


class WorklogPushError(Exception):
    """Raised for user-facing push failures."""


# ---------------------------------------------------------------------------
# .env.local reader
# ---------------------------------------------------------------------------

def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise WorklogPushError(
            f"缺少配置文件 {path}。请复制 .env.example 为 .env.local，并填写必要字段。"
        )
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Feishu doc: create or overwrite via feishu_mcp.py
# ---------------------------------------------------------------------------

def read_markdown(path: Path) -> str:
    if not path.exists():
        raise WorklogPushError(f"Markdown 文件不存在：{path}")
    if not path.is_file():
        raise WorklogPushError(f"路径不是文件：{path}")
    return path.read_text(encoding="utf-8")


def generate_doc_title() -> str:
    """生成文档标题：OpenVTER 研发日志 YYYY-MM-DD"""
    return f"OpenVTER 研发日志 {datetime.now().strftime('%Y-%m-%d')}"


def run_feishu_mcp(args: list[str], env: dict[str, str]) -> dict:
    """Run feishu_mcp.py and return parsed JSON-RPC result."""
    proc = subprocess.run(
        [sys.executable, str(FEISHU_MCP)] + args,
        capture_output=True, text=True, env=env, timeout=30,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip().split("\n")[-1]  # last line usually has the error
        raise WorklogPushError(f"feishu_mcp.py 执行失败：{stderr or '未知错误'}")
    # Parse JSON-RPC wrapper
    try:
        data = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise WorklogPushError(f"feishu_mcp.py 返回非 JSON：{exc}") from exc
    return data


def extract_result(data: dict) -> str:
    """Extract the inner result text from a JSON-RPC response."""
    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "")
    return ""


def create_feishu_doc(markdown: str, env: dict[str, str]) -> dict:
    """Create a new Feishu doc and return parsed result dict."""
    title = generate_doc_title()
    folder = env.get("FEISHU_FOLDER_TOKEN", "")
    location = {}
    if folder:
        location = {"location": folder}

    payload = {
        "title": title,
        "markdown": markdown,
    }
    if location:
        payload["location"] = location["location"]

    data = run_feishu_mcp(["create-doc", title, markdown, json.dumps(location)], env)
    inner = json.loads(extract_result(data))
    doc_id = inner.get("doc_id", "")
    doc_url = inner.get("doc_url", "")
    return {"doc_id": doc_id, "doc_url": doc_url, "message": inner.get("message", "")}


def overwrite_feishu_doc(doc_id: str, markdown: str, env: dict[str, str]) -> dict:
    """Overwrite an existing Feishu doc with new markdown content."""
    payload = json.dumps({
        "doc_id": doc_id,
        "mode": "overwrite",
        "markdown": markdown,
    })
    data = run_feishu_mcp(["update-doc", payload], env)
    inner = json.loads(extract_result(data))
    return inner


def append_feishu_doc(doc_id: str, markdown: str, env: dict[str, str]) -> dict:
    """Append content to an existing Feishu doc (prepend separator line)."""
    payload = json.dumps({
        "doc_id": doc_id,
        "mode": "append",
        "markdown": markdown,
    })
    data = run_feishu_mcp(["update-doc", payload], env)
    inner = json.loads(extract_result(data))
    return inner


# ---------------------------------------------------------------------------
# Webhook notification
# ---------------------------------------------------------------------------

def send_webhook_notification(webhook_url: str, doc_url: str, doc_id: str) -> bool:
    """Send a short notification to the group chat with doc link."""
    title = generate_doc_title()
    text = (
        f"{KEYWORD_PREFIX}"
        f"📄 {title} 已同步\n\n"
        f"👉 {doc_url}"
    )
    payload = {
        "msg_type": "text",
        "content": {"text": text},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        # Webhook failure is non-fatal (doc already created)
        print(f"警告：群聊通知发送失败 — {exc}", file=sys.stderr)
        return False

    code = result.get("code", -1)
    if code != 0:
        print(f"警告：群聊通知发送失败 — {result.get('msg', '')}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Save/load doc_id for subsequent overwrites
# ---------------------------------------------------------------------------

def save_doc_id_to_env(project_root: Path, doc_id: str) -> None:
    """Append or update FEISHU_DOC_ID in .env.local."""
    env_path = project_root / ENV_FILE
    lines = env_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found = False
    for line in lines:
        if line.startswith("FEISHU_DOC_ID="):
            new_lines.append(f"FEISHU_DOC_ID={doc_id}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"FEISHU_DOC_ID={doc_id}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync .ai-worklog/latest.md to Feishu doc + group notification."
    )
    parser.add_argument(
        "markdown", type=Path,
        help="Markdown worklog path, e.g. .ai-worklog/latest.md",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(),
        help="Project root containing .env.local. Defaults to current working directory.",
    )
    parser.add_argument(
        "--no-notify", action="store_true",
        help="Skip the group chat webhook notification.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    markdown_path = args.markdown
    if not markdown_path.is_absolute():
        markdown_path = project_root / markdown_path

    try:
        # 1. Read worklog
        content = read_markdown(markdown_path)

        # 2. Load env
        values = parse_dotenv(project_root / ENV_FILE)
        env = {
            "FEISHU_APP_ID": values.get("FEISHU_APP_ID", ""),
            "FEISHU_APP_SECRET": values.get("FEISHU_APP_SECRET", ""),
        }
        webhook_url = values.get("FEISHU_WEBHOOK_URL", "")
        folder_token = values.get("FEISHU_FOLDER_TOKEN", "")
        existing_doc_id = values.get("FEISHU_DOC_ID", "")

        if folder_token:
            env["FEISHU_FOLDER_TOKEN"] = folder_token

        if not env["FEISHU_APP_ID"] or not env["FEISHU_APP_SECRET"]:
            raise WorklogPushError("缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET，请检查 .env.local。")

        # 3. Append to existing Feishu doc (not overwrite)
        if existing_doc_id:
            doc_id = existing_doc_id
            print(f"追加到已有文档：{doc_id}")
            # Prepend separator for readability between rounds
            separator = "\n\n---\n\n"
            result = append_feishu_doc(doc_id, separator + content, {**os.environ, **env})
            doc_url = f"https://www.feishu.cn/docx/{doc_id}"
            print(f"文档追加成功：{doc_url}")
        else:
            print("创建新文档...")
            result = create_feishu_doc(content, {**os.environ, **env})
            doc_id = result["doc_id"]
            doc_url = result["doc_url"]
            print(f"文档创建成功：{doc_url}")
            save_doc_id_to_env(project_root, doc_id)
            print(f"已保存 FEISHU_DOC_ID={doc_id} 到 .env.local，后续将覆写同一篇文档")

        # 4. Send webhook notification
        if not args.no_notify and webhook_url:
            ok = send_webhook_notification(webhook_url, doc_url, doc_id)
            if ok:
                print("群聊通知发送成功")
        elif args.no_notify:
            print("（跳过群聊通知）")
        else:
            print("提示：未配置 FEISHU_WEBHOOK_URL，跳过群聊通知")

        print(f"\n📄 飞书文档: {doc_url}")

    except WorklogPushError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("错误：feishu_mcp.py 执行超时", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
