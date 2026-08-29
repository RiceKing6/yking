"""LLM 客户端封装：任何 OpenAI 兼容接口都能用（GLM / DeepSeek / Qwen / Kimi / OpenAI / Ollama…）。

支持流式输出：chat() 传入 on_text_delta 回调且 self.stream 开启时，内容增量实时回调；
工具调用参数在流中按 index 分片到达，这里负责拼装，对外仍一次性返回完整结构。
"""
from __future__ import annotations

import time
from typing import Callable

import openai
from openai import OpenAI

from .config import Config

# 这几类错误通常等一会儿重试就能过
_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

TextDeltaCallback = Callable[[str], None]


class LLMClient:
    """chat.completions 的薄封装：有限重试 + 可选流式，响应统一整理成普通 dict。

    对外只暴露一个方法：
        chat(messages, tools=None, on_text_delta=None)
            -> {"content": str|None, "tool_calls": [{"id","name","arguments"}], "usage": {...}}
    这样 agent 层和测试（FakeLLM）都不依赖 SDK 的具体类型。
    """

    def __init__(self, cfg: Config, max_retries: int = 2):
        self.model = cfg.model
        self.stream = getattr(cfg, "stream", True)
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=cfg.api_key or "EMPTY",  # Ollama 等本地服务不校验 key
            base_url=cfg.base_url,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_text_delta: TextDeltaCallback | None = None) -> dict:
        if not (self.stream and on_text_delta is not None):
            return self._chat_once(messages, tools, stream=False)
        try:
            return self._chat_once(messages, tools, stream=True, on_text_delta=on_text_delta)
        except openai.BadRequestError:
            # 个别 OpenAI 兼容端点不支持 stream_options：去掉该参数再试一次
            return self._chat_once(messages, tools, stream=True,
                                   on_text_delta=on_text_delta, include_usage=False)

    # ------------------------------------------------------------------
    def _chat_once(self, messages, tools, stream, on_text_delta=None,
                   include_usage=True) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "tools": tools or None}
        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    if include_usage:
                        kwargs["stream_options"] = {"include_usage": True}
                    resp = self.client.chat.completions.create(stream=True, **kwargs)
                    return self._consume_stream(resp, on_text_delta)
                resp = self.client.chat.completions.create(**kwargs)
                break
            except _RETRYABLE as e:
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"请求失败（已重试 {self.max_retries} 次）: {e}") from e

        msg = resp.choices[0].message
        tool_calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            tool_calls.append({
                "id": tc.id or f"call_{i}",
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            })
        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return {"content": msg.content, "tool_calls": tool_calls, "usage": usage}

    @staticmethod
    def _consume_stream(stream_resp, on_text_delta: TextDeltaCallback | None) -> dict:
        """消费 SDK 流：内容增量实时回调；工具调用分片按 index 拼装；usage 取自末尾分块。"""
        content_parts: list[str] = []
        acc: dict[int, dict] = {}
        usage = None
        for chunk in stream_resp:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = chunk.choices or []
            if not choices:
                continue  # include_usage 时最后一个分块没有 choices
            delta = choices[0].delta
            if delta is None:
                continue
            text = delta.content
            if text:
                content_parts.append(text)
                if on_text_delta is not None:
                    on_text_delta(text)
            for tc in delta.tool_calls or []:
                idx = tc.index if tc.index is not None else 0
                slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = tc.function
                if fn is not None:
                    if fn.name:
                        slot["name"] += fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments
        tool_calls = []
        for i, idx in enumerate(sorted(acc)):
            slot = acc[idx]
            tool_calls.append({
                "id": slot["id"] or f"call_{i}",
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            })
        usage_dict = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        return {"content": "".join(content_parts) or None,
                "tool_calls": tool_calls, "usage": usage_dict}
