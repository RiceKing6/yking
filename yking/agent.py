"""ReAct 循环核心：思考 -> 行动（工具调用）-> 观察 -> … -> 最终回答。"""
from __future__ import annotations

import copy
import datetime
import json
import platform
from typing import Callable

from .tools import Tool, ToolContext, execute_tool

StepCallback = Callable[[dict], None]


def build_system_prompt(ctx: ToolContext) -> str:
    today = datetime.date.today().isoformat()
    return f"""你是 yking，一个运行在命令行里的编码助手，专注于写代码、改代码，也能回答技术问题和日常提问。

# 环境
- 工作目录: {ctx.workspace}
- 操作系统: {platform.system()} {platform.release()}
- 当前日期: {today}

# 工作方式（ReAct：思考 -> 调用工具 -> 观察结果）
1. 每一步先想清楚要做什么；需要信息就调用工具获取，根据工具返回的观察结果决定下一步。
2. 信息足够后，直接给出最终回答，不再调用任何工具。
3. 修改代码前必须先用 read_file 查看目标文件的相关部分，基于真实内容修改，绝不凭空猜测。
4. 小改动优先用 edit_file 精确替换；old_string 必须与文件内容完全一致（含缩进、换行），并保证唯一。
5. 新建文件或整体重写用 write_file；浏览目录用 list_dir；全局搜代码用 grep_search。
6. 需要运行测试、脚本、pip、git 等命令时用 run_command，命令会在上述工作目录中执行。
7. 危险操作（批量删除、git push、修改系统配置等）必须先征求用户同意。
8. 用中文与用户交流；代码、命令、标识符、文件路径保持原样。
9. 任务完成后简要总结：改了哪些文件、如何验证；被卡住时如实说明原因并求助。
"""


class Agent:
    """维护对话历史并驱动 ReAct 循环。"""

    def __init__(
        self,
        llm,                        # 只要求实现 chat(messages, tools, on_text_delta) 接口
        registry: dict[str, Tool],
        ctx: ToolContext,
        system_prompt: str,
        max_steps: int = 30,
        on_event: StepCallback | None = None,
        compressor=None,            # ContextCompressor；None 表示不做上下文压缩
        memory=None,                # LongTermMemory；None 表示无长期记忆
    ):
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.max_steps = max_steps
        self.on_event = on_event
        self.compressor = compressor
        self.memory = memory
        self.last_prompt_tokens = 0  # 最近一次请求的真实 prompt tokens，用于触发压缩
        self.base_system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]

    # -- 记忆 ----------------------------------------------------------
    def _system_prompt(self) -> str:
        if self.memory is not None:
            section = self.memory.render_section()
            if section:
                return f"{self.base_system_prompt}\n\n{section}"
        return self.base_system_prompt

    def _sync_memory(self) -> None:
        """有新写入的长期记忆时，注入 system prompt 并提示一次。"""
        if self.memory is not None and self.memory.dirty:
            self.refresh_system_prompt()
            self.memory.dirty = False
            self._emit({"kind": "memory_saved", "count": len(self.memory.entries)})

    def refresh_system_prompt(self) -> None:
        """按当前长期记忆重建 system prompt（记忆可能被其他会话/子 Agent 更新过）。"""
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._system_prompt()

    def reset(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self.last_prompt_tokens = 0

    def run_turn(self, user_input: str) -> str:
        """处理用户一轮输入：循环调用模型与工具，返回最终回答文本。"""
        self._sync_memory()
        snapshot = copy.deepcopy(self.messages)  # 出错/中断时回滚，保证消息序列完整
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop()
        except BaseException:
            self.messages = snapshot
            raise

    # ------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        """向展示层（CLI）发事件；展示层出错不影响 agent 本身。"""
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.registry.values()]

    def _loop(self) -> str:
        for step in range(1, self.max_steps + 1):
            self._emit({"kind": "step", "step": step})
            if self.compressor is not None:
                compressed = self.compressor.maybe_compress(
                    self.messages, self.last_prompt_tokens)
                if compressed is not None:
                    self.last_prompt_tokens = compressed
            state = {"streamed": False}

            def on_text_delta(delta: str) -> None:
                state["streamed"] = True
                self._emit({"kind": "text_delta", "delta": delta})

            reply = self.llm.chat(self.messages, self._tool_schemas(),
                                  on_text_delta=on_text_delta)
            usage = reply.get("usage") or {}
            if usage.get("prompt_tokens"):
                self.last_prompt_tokens = usage["prompt_tokens"]
            if usage:
                self._emit({"kind": "usage", **usage})

            content = reply.get("content")
            tool_calls = reply.get("tool_calls") or []

            # 模型没有调用工具 => 最终回答，本轮结束
            if not tool_calls:
                text = content or "（模型没有返回内容）"
                self.messages.append({"role": "assistant", "content": text})
                self._emit({"kind": "final", "text": text})
                return text

            # 有工具调用：先记录 assistant 消息，再逐个执行并追加观察结果
            self.messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            })
            # 流式模式下内容已经实时展示过，不再整段重发
            if content and not state["streamed"]:
                self._emit({"kind": "text", "text": content})

            for tc in tool_calls:
                name = tc["name"]
                raw_args = tc["arguments"]
                self._emit({"kind": "tool_call", "name": name, "arguments": raw_args})
                try:
                    args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                    if not isinstance(args, dict):
                        raise ValueError("参数必须是 JSON 对象")
                except (json.JSONDecodeError, ValueError) as e:
                    result = f"错误: 工具参数不是合法的 JSON 对象（{e}）: {raw_args[:200]}"
                else:
                    result = execute_tool(self.registry, self.ctx, name, args)
                self._emit({"kind": "tool_result", "name": name, "result": result})
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
                self._sync_memory()

        text = (f"已连续执行 {self.max_steps} 步仍未给出最终回答，本轮先停在这里。"
                "你可以继续对话让我接着做。")
        self.messages.append({"role": "assistant", "content": text})
        self._emit({"kind": "final", "text": text})
        return text
