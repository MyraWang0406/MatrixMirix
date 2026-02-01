"""封装 OpenRouter chat completions 调用"""
from __future__ import annotations

import json
import os
from typing import Any, Tuple, Union

import httpx
# 本地可用 .env，云端依赖环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://openrouter.ai/api/v1"

RETRY_MESSAGE = (
    "你上次输出不合法，请只输出合法JSON。"
    "直接输出纯 JSON（对象或数组），不要用 Markdown 代码块包裹，不要输出任何解释文字。"
)


class JsonParseError(Exception):
    """JSON 解析失败（含重试后仍失败）"""

    def __init__(self, message: str, raw_content: str = ""):
        super().__init__(message)
        self.raw_content = raw_content


def _get_api_key() -> str:
    """优先从环境变量读取，云端必须配置 OPENROUTER_API_KEY"""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "请设置环境变量 OPENROUTER_API_KEY（Streamlit Cloud：Settings → Secrets）"
        )
    return key


def _get_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    调用 OpenRouter chat completions API，返回 assistant 的 content 文本。
    """
    api_key = _get_api_key()
    model = model or _get_model()

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://creative-eval-demo.local",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("OpenRouter 返回空内容")
    return content.strip()


def _strip_markdown_fences(s: str) -> str:
    """去掉 ``` 或 ```json 包裹（如果有）"""
    t = s.strip()
    if t.startswith("```"):
        # 可能是 ```json\n...\n```
        # 找第一行结束
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        # 去尾部 ```
        last_fence = t.rfind("```")
        if last_fence != -1:
            t = t[:last_fence]
    return t.strip()


def _extract_json_text(content: str) -> str:
    """
    从 content 中提取 JSON 文本：
    1) 去 markdown code fence
    2) 尝试截取从第一个 '{' 或 '[' 开始的片段（处理“前面有解释文字”）
    """
    s = _strip_markdown_fences(content)

    # 处理“前面有废话，后面才是 JSON”
    obj_pos = s.find("{")
    arr_pos = s.find("[")
    if obj_pos == -1 and arr_pos == -1:
        return s.strip()

    start = obj_pos if (obj_pos != -1 and (arr_pos == -1 or obj_pos < arr_pos)) else arr_pos
    return s[start:].strip()


JsonType = Union[dict[str, Any], list[Any]]


def chat_completion_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    retry_on_parse_error: bool = True,
    return_raw: bool = False,
) -> Union[JsonType, Tuple[JsonType, str]]:
    """
    调用 chat_completion，解析返回为 JSON。

    - 若 json.loads 失败且 retry_on_parse_error=True，则重试一次并附上「请只输出合法JSON」提示；
    - 重试后仍失败则抛出 JsonParseError。
    - return_raw=True 时返回 (parsed_json, raw_content) 方便在 UI 显示 Raw Output
    """
    msgs = messages
    last_raw = ""

    for attempt in range(2):
        content = chat_completion(msgs, model=model, temperature=temperature, max_tokens=max_tokens)
        last_raw = content

        json_text = _extract_json_text(content)

        try:
            parsed = json.loads(json_text)
            return (parsed, content) if return_raw else parsed

        except json.JSONDecodeError as e:
            if attempt == 0 and retry_on_parse_error:
                msgs = list(msgs) + [{"role": "user", "content": RETRY_MESSAGE}]
                continue

            # 截断 raw，避免太长
            raw_short = content if len(content) <= 4000 else content[:4000] + "\n...<TRUNCATED>..."
            raise JsonParseError(
                f"JSON 解析失败: {e}",
                raw_content=raw_short,
            ) from e

    # 理论不会到这
    raw_short = last_raw if len(last_raw) <= 4000 else last_raw[:4000] + "\n...<TRUNCATED>..."
    raise JsonParseError("JSON 解析失败（重试后仍无效）", raw_content=raw_short)
