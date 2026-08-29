"""冒烟测试（无需 API Key）：验证工具实现与 ReAct 循环的正确性。

运行: conda run -n yking python tests/test_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yking.agent import Agent                                    # noqa: E402
from yking.tools import ToolContext, build_registry, execute_tool  # noqa: E402


class FakeLLM:
    """按脚本依次返回预设响应，用于不联网测试 ReAct 循环。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools, on_text_delta=None):
        self.calls.append((list(messages), tools))
        if not self.script:
            raise AssertionError("FakeLLM 脚本已耗尽，但 agent 仍在请求模型")
        return self.script.pop(0)


def make_ctx(root: Path) -> ToolContext:
    return ToolContext(workspace=root, auto_approve=True)


def test_file_tools(ctx: ToolContext) -> None:
    reg = build_registry(ctx)

    r = execute_tool(reg, ctx, "write_file", {"path": "src/a.py", "content": "x = 1\ny = 2\n"})
    assert r.startswith("OK"), r
    assert (ctx.workspace / "src" / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"

    r = execute_tool(reg, ctx, "read_file", {"path": "src/a.py"})
    assert "x = 1" in r and "y = 2" in r, r

    r = execute_tool(reg, ctx, "edit_file", {"path": "src/a.py",
                                             "old_string": "x = 1", "new_string": "x = 42"})
    assert r.startswith("OK"), r
    assert "x = 42" in (ctx.workspace / "src" / "a.py").read_text(encoding="utf-8")

    r = execute_tool(reg, ctx, "edit_file", {"path": "src/a.py",
                                             "old_string": "不存在的内容", "new_string": "?"})
    assert r.startswith("错误"), r

    r = execute_tool(reg, ctx, "list_dir", {})
    assert "src/" in r, r

    r = execute_tool(reg, ctx, "grep_search", {"pattern": r"x = \d+"})
    assert "a.py" in r and "x = 42" in r, r

    r = execute_tool(reg, ctx, "run_command", {"command": "echo hello_yking"})
    assert "hello_yking" in r and "exit_code=0" in r, r

    r = execute_tool(reg, ctx, "no_such_tool", {})
    assert r.startswith("错误"), r

    print("  [ok] 文件与命令工具")


def test_encoding_safety(ctx) -> None:
    """无法按 UTF-8/系统编码解码的文件必须被拒绝编辑/覆盖，字节原样保留。"""
    reg = build_registry(ctx)
    payload = b"\xff\xfe\x81\x9d" * 8  # utf-8 / cp1252 / gbk 严格解码均失败
    p = ctx.workspace / "legacy.bin"
    p.write_bytes(payload)

    # patch 掉系统编码回退，让所有平台上都确定性地"无法解码"
    with patch("yking.tools.fs.locale.getpreferredencoding", return_value="utf-8"):
        r = execute_tool(reg, ctx, "edit_file",
                         {"path": "legacy.bin", "old_string": "x", "new_string": "y"})
        assert r.startswith("错误") and "拒绝" in r, r
        assert p.read_bytes() == payload, "拒绝编辑后文件字节必须原样保留"

        r = execute_tool(reg, ctx, "write_file",
                         {"path": "legacy.bin", "content": "overwrite"})
        assert r.startswith("错误") and "拒绝" in r, r
        assert p.read_bytes() == payload, "拒绝覆盖后文件字节必须原样保留"

    # 正常 UTF-8 文件的编辑不受影响
    execute_tool(reg, ctx, "write_file", {"path": "ok.py", "content": "a = 1\n"})
    r = execute_tool(reg, ctx, "edit_file",
                     {"path": "ok.py", "old_string": "a = 1", "new_string": "a = 2"})
    assert r.startswith("OK"), r
    print("  [ok] 编码安全：无法解码的文件拒绝编辑/覆盖，正常 UTF-8 编辑不受影响")


def test_react_loop(ctx: ToolContext) -> None:
    reg = build_registry(ctx)
    script = [
        {   # 第 1 步：模型决定调用工具
            "content": "我先写入一个文件。",
            "tool_calls": [{
                "id": "call_1",
                "name": "write_file",
                "arguments": json.dumps({"path": "hello.txt", "content": "hi yking"}),
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        {   # 第 2 步：拿到观察结果后给出最终回答
            "content": "文件已写好，内容是 hi yking。",
            "tool_calls": [],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8},
        },
    ]
    llm = FakeLLM(script)
    events = []
    agent = Agent(llm, reg, ctx, system_prompt="你是测试助手",
                  max_steps=5, on_event=events.append)
    final = agent.run_turn("帮我创建 hello.txt")

    assert final == "文件已写好，内容是 hi yking。", final
    assert (ctx.workspace / "hello.txt").read_text(encoding="utf-8") == "hi yking"

    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], roles
    tool_msg = agent.messages[3]
    assert tool_msg["tool_call_id"] == "call_1" and "OK" in tool_msg["content"]

    kinds = [e["kind"] for e in events]
    assert kinds == ["step", "usage", "text", "tool_call", "tool_result",
                     "step", "usage", "final"], kinds

    # 第二轮：历史应被保留（发给模型的消息里包含上一轮的 tool 消息）
    llm.script.append({"content": "好的。", "tool_calls": [], "usage": {}})
    final2 = agent.run_turn("继续")
    assert final2 == "好的。", final2
    sent = llm.calls[-1][0]
    assert any(m.get("role") == "tool" for m in sent)

    print("  [ok] ReAct 循环（工具调用 / 观察 / 最终回答 / 历史保留）")


def test_error_rollback(ctx: ToolContext) -> None:
    reg = build_registry(ctx)

    class BoomLLM:
        def chat(self, messages, tools, on_text_delta=None):
            raise RuntimeError("网络错误")

    agent = Agent(BoomLLM(), reg, ctx, system_prompt="s", max_steps=3)
    before = [dict(m) for m in agent.messages]
    try:
        agent.run_turn("你好")
        raise AssertionError("应当抛出异常")
    except RuntimeError:
        pass
    assert agent.messages == before, "异常后消息历史应回滚"

    print("  [ok] 异常时消息历史回滚")


def main() -> int:
    print("yking 冒烟测试:")
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(Path(td))
        test_file_tools(ctx)
        test_encoding_safety(ctx)
        test_react_loop(ctx)
        test_error_rollback(ctx)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
