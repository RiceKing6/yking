"""可观测性冒烟测试（无需 API Key）：请求 ID、耗时、重试次数、累计 token 统计。

覆盖：
- LLM 层：每次 chat() 请求 ID 唯一且格式为 req-N；重试共享同一 ID 并如实记录 attempts；
  elapsed 非负；流式路径同样带观测字段。
- Agent 层：total_usage 正确累计（模型调用/工具调用/耗时/token）；
  usage 事件带 request_id/elapsed/total；tool_result 事件带工具耗时；final 事件带本轮耗时。

运行: conda run -n yking python tests/test_observability.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openai                                              # noqa: E402
from yking.agent import Agent, NEW_USAGE_TOTALS            # noqa: E402
from yking.config import Config                            # noqa: E402
from yking.llm import LLMClient                            # noqa: E402
from yking.tools import ToolContext, build_registry        # noqa: E402


class FakeCompletions:
    """脚本化的假 SDK：script 元素为响应对象或待抛异常。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if kwargs.get("stream"):
            return iter(item)
        return item


def make_client(script, stream=True, max_retries=2):
    cfg = Config(api_key="dummy", base_url="https://example.com",
                 model="test", stream=stream)
    client = LLMClient(cfg, max_retries=max_retries)
    fake = FakeCompletions(script)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def plain_response(text="回答"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    return resp


def rate_limit_error():
    return openai.RateLimitError(
        message="slow down",
        response=SimpleNamespace(status_code=429, headers={}, request=None),
        body=None,
    )


def test_request_id_and_attempts() -> None:
    client, fake = make_client([plain_response("第一次"), plain_response("第二次")],
                               stream=False)
    r1 = client.chat([{"role": "user", "content": "hi"}])
    r2 = client.chat([{"role": "user", "content": "hi"}])
    assert re.fullmatch(r"req-\d+", r1["request_id"]), r1["request_id"]
    assert r1["request_id"] != r2["request_id"], "每次 chat 应有独立请求 ID"
    assert r1["attempts"] == 1
    assert r1["elapsed"] >= 0
    print("  [ok] LLM 层：请求 ID 唯一且格式正确，attempts=1，elapsed 非负")


def test_retry_shares_request_id() -> None:
    client, fake = make_client([rate_limit_error(), plain_response("恢复")], stream=False)
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r["attempts"] == 2, f"重试后 attempts 应为 2，实际 {r['attempts']}"
    assert re.fullmatch(r"req-\d+", r["request_id"])
    assert len(fake.calls) == 2, "应真实发出两次请求"
    print("  [ok] LLM 层：重试共享同一请求 ID 且如实记录 attempts=2")


class ScriptedLLM:
    """给 Agent 用的假模型：返回带观测字段的响应。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, on_text_delta=None):
        self.calls.append(list(messages))
        return self.script.pop(0)


def obs_reply(content=None, tool_calls=None, prompt=100, completion=10,
              req="req-1", elapsed=0.5):
    return {"content": content, "tool_calls": tool_calls or [], "usage": {
        "prompt_tokens": prompt, "completion_tokens": completion},
        "request_id": req, "elapsed": elapsed, "attempts": 1}


def test_agent_accumulation() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctx = ToolContext(workspace=Path(td), auto_approve=True)
        registry = build_registry(ctx)
        llm = ScriptedLLM([
            obs_reply(content="我来写文件", req="req-a", elapsed=1.2,
                      tool_calls=[{"id": "c1", "name": "write_file",
                                   "arguments": json.dumps({"path": "a.txt", "content": "x"})}]),
            obs_reply(content="写好了", req="req-b", elapsed=0.8, prompt=200, completion=30),
        ])
        events = []
        agent = Agent(llm, registry, ctx, system_prompt="你是助手", on_event=events.append)
        agent.run_turn("写个文件")

        u = agent.total_usage
        assert u["llm_calls"] == 2, u
        assert u["prompt_tokens"] == 300 and u["completion_tokens"] == 40, u
        assert abs(u["llm_seconds"] - 2.0) < 0.01, u
        assert u["tool_calls"] == 1 and u["tool_seconds"] >= 0, u
        assert u["llm_calls"] > 0 and set(u) == set(NEW_USAGE_TOTALS), "统计字段应完整"

        # usage 事件带请求 ID / 耗时 / 累计
        usages = [e for e in events if e["kind"] == "usage"]
        assert usages[0]["request_id"] == "req-a" and usages[0]["elapsed"] == 1.2
        assert usages[1]["total"]["llm_calls"] == 2
        assert usages[1]["total"]["prompt_tokens"] == 300

        # 工具结果事件带耗时
        tr = next(e for e in events if e["kind"] == "tool_result")
        assert tr["elapsed"] is not None and tr["elapsed"] >= 0

        # final 事件带本轮总耗时
        fin = next(e for e in events if e["kind"] == "final")
        assert fin["elapsed"] >= 0

        # reset（/clear）不清零累计统计
        before = dict(agent.total_usage)
        agent.reset()
        assert agent.total_usage == before, "reset 不应复位累计观测数据"
        print("  [ok] Agent 层：累计 token/调用/耗时正确，事件携带观测字段，reset 不复位统计")


def main() -> int:
    print("yking 可观测性冒烟测试:")
    test_request_id_and_attempts()
    test_retry_shares_request_id()
    test_agent_accumulation()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
