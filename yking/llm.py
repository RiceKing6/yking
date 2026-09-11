"""LLM 客户端封装：任何 OpenAI 兼容接口都能用（GLM / DeepSeek / Qwen / Kimi / OpenAI / Ollama…）。

支持流式输出：chat() 传入 on_text_delta 回调且 self.stream 开启时，内容增量实时回调；
工具调用参数在流中按 index 分片到达，这里负责拼装，对外仍一次性返回完整结构。

可观测性：每次 chat() 生成一个请求 ID（同一逻辑请求的重试共享同一个 ID），
并返回本次调用的墙钟耗时与真实请求次数，供上层记录与展示。
"""
from __future__ import annotations

import itertools
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

# 进程内单调递增的请求序号；CPython 下 next() 是原子的，多线程安全
_REQUEST_COUNTER = itertools.count(1)


class LLMClient:
    """chat.completions 的薄封装：有限重试 + 可选流式，响应统一整理成普通 dict。

    对外只暴露一个方法：
        chat(messages, tools=None, on_text_delta=None)
            -> {"content": str|None, "tool_calls": [{"id","name","arguments"}], "usage": {...},
                "request_id": "req-N", "elapsed": float, "attempts": int}
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
        request_id = f"req-{next(_REQUEST_COUNTER)}"
        started = time.monotonic()
        stats = {"attempts": 0}
        if not (self.stream and on_text_delta is not None):
            reply = self._chat_once(messages, tools, stream=False, stats=stats)
        else:
            try:
                reply = self._chat_once(messages, tools, stream=True,
                                        on_text_delta=on_text_delta, stats=stats)
            except openai.BadRequestError:
                # 个别 OpenAI 兼容端点不支持 stream_options：去掉该参数再试一次
                reply = self._chat_once(messages, tools, stream=True,
                                        on_text_delta=on_text_delta,
                                        include_usage=False, stats=stats)
        reply["request_id"] = request_id
        reply["elapsed"] = time.monotonic() - started
        reply["attempts"] = stats["attempts"]
        return reply

    # ------------------------------------------------------------------
    def _chat_once(self, messages, tools, stream, on_text_delta=None,
                   include_usage=True, stats=None) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "tools": tools or None}
        if stream:
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
            resp = self._create_with_retry(kwargs, stream=True, stats=stats)
            # 流一旦开始消费就不再重试：中途网络断开时已回调的增量无法撤回，
            # 重试会让内容重复输出。直接抛错，由上层回滚本轮对话。
            return self._consume_stream(resp, on_text_delta)

        resp = self._create_with_retry(kwargs, stream=False, stats=stats)
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

    def _create_with_retry(self, kwargs: dict, stream: bool, stats: dict | None = None):
        """只对 create() 做有限重试（限流/连接/5xx）；流式内容的消费不在重试范围内。

        stats["attempts"] 记录真实发出的请求次数（含重试），供可观测性使用。
        """
        for attempt in range(self.max_retries + 1):
            if stats is not None:
                stats["attempts"] = stats.get("attempts", 0) + 1
            try:
                if stream:
                    return self.client.chat.completions.create(stream=True, **kwargs)
                return self.client.chat.completions.create(**kwargs)
            except _RETRYABLE as e:
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"请求失败（已重试 {self.max_retries} 次）: {e}") from e

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
