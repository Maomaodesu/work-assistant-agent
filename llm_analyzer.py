"""
LLM 分析器：调用 AMD Radeon API，对中断快照生成工作摘要。
"""

import json
from pathlib import Path

from settings import get_api_key, get_settings
from streaming_runtime import stream_chat_completion_text

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装：pip install openai")

def _load_api_key(key_file: str | None = None) -> str:
    if key_file:
        return Path(key_file).read_text(encoding="utf-8").strip()
    return get_api_key()


def _load_prompt(prompt_file: str = "interrupt_analysis_prompt.txt") -> str:
    path = Path(__file__).parent / prompt_file
    return path.read_text(encoding="utf-8")


def analyze_snapshot(
    snapshot: dict,
    model: str | None = None,
    key_file: str | None = None,
) -> str:
    """
    将快照 JSON 发送给 AMD Radeon API，返回中断摘要文本。
    """
    settings = get_settings()
    api_key    = _load_api_key(key_file)
    model = model or settings.amd_model
    sys_prompt = _load_prompt()

    # 裁剪快照：diff 内容只保留前50行，避免超出 token 限制
    snapshot_trimmed = json.loads(json.dumps(snapshot))  # 深拷贝
    for proj in snapshot_trimmed.get("projects", []):
        if not settings.send_commit_info:
            proj["commits"] = []
        for diff_key in ("diff_unstaged", "diff_staged"):
            if not settings.send_diff_summary:
                proj[diff_key] = ""
                continue
            text = proj.get(diff_key, "")
            if text:
                lines = text.splitlines()
                proj[diff_key] = "\n".join(lines[:50]) + (
                    f"\n...（已截断，共{len(lines)}行）" if len(lines) > 50 else ""
                )

    user_content = (
        "以下是工作快照 JSON，请按照系统提示的格式生成中断摘要：\n\n"
        + json.dumps(snapshot_trimmed, ensure_ascii=False, indent=2)
    )

    client = OpenAI(
        api_key=api_key,
        base_url=settings.amd_base_url,
    )

    return stream_chat_completion_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        max_tokens=2048,
        timeout=settings.request_timeout_seconds,
    )
