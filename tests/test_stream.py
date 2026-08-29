"""流式输出（streaming）单元测试：用假的 SDK 流验证增量回调与内容/工具调用拼装。

无需 API Key。
运行: conda run -n yking python tests/test_stream.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openai                                                       # noqa: E402
from yking.config import Config                                     # noqa: E402
from yking.llm import LLMClient                                     # noqa: E402


def make_chunk(content=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def make_tc(index, tc_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=fn)


class FakeCompletions:
    """按脚本返回假流 / 假完整响应，并记录每次调用的参数。"""

    def __init__(self, chunks, responses=None):
        self.chunks = list(chunks)
        self.responses = list(responses or [])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self.responses.pop(0)


def make_client(chunks=(), responses=(), stream=True):
    cfg = Config(api_key="dummy", base_url="https://example.com",
                 model="test", stream=stream)
    client = LLMClient(cfg)
    fake = FakeCompletions(chunks, responses)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_stream_assembly() -> None:
    """内容增量实时回调；跨分片的工具调用参数按 index 拼装；usage 取自末尾分块。"""
    chunks = [
        make_chunk(content="你好"),
        make_chunk(tool_calls=[make_tc(0, tc_id="call_1",
                                       name="write_file", arguments='{"pa')]),
        make_chunk(content="世界"),
        make_chunk(tool_calls=[make_tc(0, arguments='th": "a.py"}')]),
        make_chunk(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)),
    ]
    client, fake = make_client(chunks)
    deltas = []
    reply = client.chat([{"role": "user", "content": "hi"}], on_text_delta=deltas.append)

    assert deltas == ["你好", "世界"], deltas
    assert reply["content"] == "你好世界"
    assert reply["tool_calls"] == [{"id": "call_1", "name": "write_file",
                                    "arguments": '{"path": "a.py"}'}]
    assert reply["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["stream_options"] == {"include_usage": True}
    print("  [ok] 流式：增量回调 + 内容/工具调用/usage 拼装")


def test_stream_disabled_ignores_callback() -> None:
    msg = SimpleNamespace(content="最终回答", tool_calls=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    client, fake = make_client(responses=[resp], stream=False)
    deltas = []
    reply = client.chat([{"role": "user", "content": "hi"}], on_text_delta=deltas.append)

    assert deltas == []
    assert reply["content"] == "最终回答"
    assert fake.calls[0].get("stream") is None
    print("  [ok] stream=False 时忽略回调、走非流式")


def test_stream_without_usage_option_fallback() -> None:
    """端点不支持 stream_options（BadRequest）→ 自动去掉该参数重试一次。"""
    chunks = [make_chunk(content="ok")]
    client, fake = make_client(chunks)

    real_create = fake.create
    calls = []

    def create(**kwargs):
        calls.append(kwargs)  # 失败的调用也记录（FakeCompletions 只记录成功的）
        if kwargs.get("stream_options"):
            raise openai.BadRequestError(
                message="stream_options is not supported",
                response=SimpleNamespace(status_code=400, headers={}, request=None),
                body=None,
            )
        return real_create(**kwargs)

    fake.create = create
    deltas = []
    reply = client.chat([{"role": "user", "content": "hi"}], on_text_delta=deltas.append)

    assert deltas == ["ok"]
    assert reply["content"] == "ok"
    assert reply["usage"] == {}
    assert calls[0].get("stream_options") is not None
    assert "stream_options" not in calls[1]
    print("  [ok] 不支持 stream_options 的端点自动降级重试")


def test_no_retry_mid_stream() -> None:
    """流开始消费后不再重试：中途断线直接抛错，已显示的增量不会重复输出。"""

    def flaky_stream():
        yield make_chunk(content="你好")
        # 模拟流传输中途断线（可重试类错误）
        raise openai.APIConnectionError(request=SimpleNamespace())

    client, fake = make_client()

    def create(**kwargs):
        fake.calls.append(kwargs)
        return flaky_stream()

    fake.create = create
    deltas = []
    try:
        client.chat([{"role": "user", "content": "hi"}], on_text_delta=deltas.append)
        raise AssertionError("应当抛出连接错误")
    except openai.APIConnectionError:
        pass
    assert deltas == ["你好"], deltas
    assert len(fake.calls) == 1, "流开始消费后不得重试"
    print("  [ok] 流式中途断线：不重试、不重复输出，直接抛错交由上层回滚")


def main() -> int:
    print("yking streaming 冒烟测试:")
    test_stream_assembly()
    test_stream_disabled_ignores_callback()
    test_stream_without_usage_option_fallback()
    test_no_retry_mid_stream()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
