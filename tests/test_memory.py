"""记忆系统冒烟测试（无需 API Key）。

覆盖：长期记忆（保存/去重/持久化/删除/淘汰/注入渲染）、token 估算、
上下文压缩（机械裁剪保结构 / LLM 摘要替换 / 安全边界）、Agent 集成
（save_memory 工具 → system prompt 刷新 → 落盘；压缩在循环内真实触发）。

运行: conda run -n yking python tests/test_memory.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yking.agent import Agent                                       # noqa: E402
from yking.memory import ContextCompressor, LongTermMemory, estimate_tokens  # noqa: E402
from yking.tools import ToolContext, build_registry                 # noqa: E402


class FakeLLM:
    """按脚本依次返回预设响应，并记录每次请求。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, on_text_delta=None):
        self.calls.append((list(messages), tools))
        if not self.script:
            raise AssertionError("FakeLLM 脚本已耗尽")
        return self.script.pop(0)


class NoLLM:
    """压缩只允许走机械裁剪时的守卫：一旦被调用就让测试失败。"""

    def chat(self, messages, tools, on_text_delta=None):
        raise AssertionError("机械裁剪阶段不应调用 LLM")


def make_ctx(root: Path, memory=None) -> ToolContext:
    return ToolContext(workspace=root, auto_approve=True, memory=memory)


def test_long_term_memory(tmp: Path) -> None:
    path = tmp / "mem.json"
    mem = LongTermMemory(path)
    assert mem.save("用户偏好使用中文回复").startswith("OK")
    assert "已存在" in mem.save("用户偏好使用中文回复")  # 去重（忽略大小写）
    mem.save("项目测试用 pytest")

    mem2 = LongTermMemory(path)  # 新实例 = 新会话，应从磁盘恢复
    assert "pytest" in mem2.list() and "中文回复" in mem2.list()

    r = mem2.forget("1")  # 按编号删除
    assert "已删除" in r and "用户偏好使用中文回复" in r  # 确认消息回显被删内容
    mem3 = LongTermMemory(path)
    assert "中文回复" not in mem3.list() and "pytest" in mem3.list()

    r = mem3.forget("不存在的词")
    assert r.startswith("未找到")

    mem4 = LongTermMemory(tmp / "mem4.json", max_entries=2)
    mem4.save("a"); mem4.save("b")
    r = mem4.save("c")
    assert "最旧" in r and len(mem4.entries) == 2 and mem4.entries[0]["content"] == "b"

    assert "长期记忆" in mem4.render_section()
    assert LongTermMemory(tmp / "empty.json").render_section() == ""
    print("  [ok] 长期记忆：保存/去重/持久化/编号删除/关键词删除/容量淘汰/渲染")


def test_estimate_tokens() -> None:
    assert estimate_tokens([{"role": "user", "content": "你好世界"}]) >= 4
    assert estimate_tokens([{"role": "user", "content": "abcdefgh"}]) >= 2
    assert estimate_tokens([{"role": "user", "content": ""}]) == 0
    print("  [ok] token 粗估（CJK 与 ASCII 区分对待）")


def test_mechanical_trim_only(tmp: Path) -> None:
    """预算超限但找不到安全摘要边界 → 只做机械裁剪，消息结构与配对原样保留。"""
    big = "x" * 2000
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "第一轮需求"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": big},
        {"role": "assistant", "content": "第一轮完成"},
        {"role": "user", "content": "第二轮"},
    ]
    events = []
    comp = ContextCompressor(NoLLM(), budget_tokens=50, trigger_ratio=0.5,
                             keep_recent=2, on_event=events.append)
    result = comp.maybe_compress(messages, last_prompt_tokens=0)

    assert result is not None
    assert len(messages) == 6, "机械裁剪不改变消息数量"
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert "工具输出已压缩" in messages[3]["content"]
    assert len(messages[3]["content"]) < 500
    assert messages[3]["tool_call_id"] == "c1", "tool 消息的配对 id 必须保留"
    assert messages[3]["content"].endswith(big[-100:]), "保留尾部"
    kinds = [(e["kind"], e.get("summary")) for e in events]
    assert ("compress", False) in kinds
    print("  [ok] 机械裁剪：超长工具输出被截断，结构/配对/数量不变，未调 LLM")


def test_summary_stage(tmp: Path) -> None:
    """机械裁剪后仍超预算 → 找到 user 轮次边界，把更早对话摘要成一条消息。"""
    script = [{"content": "摘要：用户要求创建文件，已写 big.txt 完成。",
               "tool_calls": [], "usage": {}}]
    llm = FakeLLM(script)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "第一轮需求"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 2000},
        {"role": "assistant", "content": "第一轮完成"},
        {"role": "user", "content": "第二轮"},
    ]
    events = []
    comp = ContextCompressor(llm, budget_tokens=50, trigger_ratio=0.5,
                             keep_recent=2, on_event=events.append)
    comp.maybe_compress(messages, last_prompt_tokens=0)

    assert len(messages) == 3, messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user" and "摘要" in messages[1]["content"]
    assert "big.txt" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "第二轮"}, "当前轮消息必须保留"
    # 摘要请求里应包含被摘要段的内容
    sent = llm.calls[0][0]
    assert any("第一轮需求" in (m.get("content") or "") for m in sent)
    assert [e["kind"] for e in events] == ["compress"]
    assert events[0]["summary"] is True
    print("  [ok] LLM 摘要：早期对话替换为摘要消息，当前轮完整保留")


def test_agent_memory_integration(tmp: Path) -> None:
    memory = LongTermMemory(tmp / "m.json")
    ctx = make_ctx(tmp, memory=memory)
    registry = build_registry(ctx)
    assert "save_memory" in registry, "启用记忆后应注册 save_memory 工具"

    llm = FakeLLM([
        {"content": None, "usage": {"prompt_tokens": 100, "completion_tokens": 5},
         "tool_calls": [{"id": "c1", "name": "save_memory",
                         "arguments": json.dumps({"content": "用户偏好深色主题"},
                                                 ensure_ascii=False)}]},
        {"content": "已记住你的偏好。", "tool_calls": [],
         "usage": {"prompt_tokens": 150, "completion_tokens": 5}},
    ])
    events = []
    agent = Agent(llm, registry, ctx, system_prompt="你是助手", memory=memory,
                  on_event=events.append)
    final = agent.run_turn("记住我喜欢深色主题")

    assert "已记住" in final
    assert memory.entries and "深色主题" in memory.entries[0]["content"]
    assert "深色主题" in agent.messages[0]["content"], "system prompt 应已刷新"
    assert any(e["kind"] == "memory_saved" for e in events)
    assert (tmp / "m.json").exists(), "记忆应落盘"

    # 第二轮：发出的请求里 system prompt 携带长期记忆
    llm.script.append({"content": "好的。", "tool_calls": [], "usage": {}})
    agent.run_turn("继续")
    sent_system = llm.calls[-1][0][0]
    assert sent_system["role"] == "system" and "深色主题" in sent_system["content"]

    # 未启用记忆时不应注册工具
    assert "save_memory" not in build_registry(make_ctx(tmp))
    print("  [ok] Agent 集成：save_memory 工具 → 落盘 → system prompt 刷新 → 跨轮可见")


def test_agent_compression_integration(tmp: Path) -> None:
    ctx = make_ctx(tmp)
    registry = build_registry(ctx)
    llm = FakeLLM([
        {"content": "我来写文件", "usage": {"prompt_tokens": 9999, "completion_tokens": 5},
         "tool_calls": [{"id": "c1", "name": "write_file",
                         "arguments": json.dumps({"path": "big.txt", "content": "x" * 500})}]},
        {"content": "第一轮完成", "tool_calls": [],
         "usage": {"prompt_tokens": 9999, "completion_tokens": 5}},
        {"content": "摘要：用户让我写大文件，已写 big.txt。", "tool_calls": [], "usage": {}},
        {"content": "第二轮完成", "tool_calls": [], "usage": {}},
    ])
    events = []
    compressor = ContextCompressor(llm, budget_tokens=50, trigger_ratio=0.5,
                                   keep_recent=3, on_event=events.append)
    agent = Agent(llm, registry, ctx, system_prompt="你是助手",
                  compressor=compressor, on_event=events.append)
    agent.run_turn("写个大文件")
    agent.run_turn("第二轮")

    kinds = [e["kind"] for e in events]
    assert "compress" in kinds
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "user", "assistant"], roles
    assert "摘要" in agent.messages[1]["content"]
    assert agent.messages[-1]["content"] == "第二轮完成"
    print("  [ok] Agent 集成：多轮对话中压缩真实触发，历史被摘要替换")


def main() -> int:
    print("yking 记忆系统冒烟测试:")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_long_term_memory(tmp)
        test_estimate_tokens()
        test_mechanical_trim_only(tmp)
        test_summary_stage(tmp)
        test_agent_memory_integration(tmp)
        test_agent_compression_integration(tmp)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
