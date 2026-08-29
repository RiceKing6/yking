"""记忆系统：短期记忆（消息历史）+ 上下文压缩 + 长期记忆（跨会话落盘）。

- 短期记忆即 Agent.messages（会话内消息历史），由 ContextCompressor 在逼近
  token 预算时自动压缩。
- ContextCompressor 两级策略：
  1) 机械裁剪（不调 LLM）：截断较早消息里的超长工具输出/文本，消息结构与顺序原样保留；
  2) LLM 摘要：仍超预算时，把更早的一段对话整体摘要成一条消息替换。
     切分点只选在 user 消息处（轮次边界），绝不拆散 assistant(tool_calls)
     与其 tool 结果的配对，保证压缩后的消息序列仍然合法。
- LongTermMemory：跨会话长期记忆，JSON 落盘在工作目录 .yking/memory.json，
  内容注入 system prompt（Agent 在记忆变化时自动刷新）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

EventCallback = Callable[[dict], None]

SUMMARIZE_SYSTEM = (
    "你是会话摘要器。把给定的历史对话压缩成一份简洁的中文摘要，作为后续对话的背景知识。"
    "必须保留：用户的目标与要求、已做的关键决定、改动过的文件与结果、重要事实与未完成事项。"
    "不要寒暄和评论，直接输出摘要正文。"
)


def estimate_tokens(messages: list[dict]) -> int:
    """粗略估算消息序列的 token 数：CJK 字符约 1 token/字，其他约 4 字符/token。"""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if not content:
            continue
        cjk = sum(1 for ch in content if ord(ch) > 0x2E80)
        total += cjk + (len(content) - cjk) // 4 + 8  # 每条消息的固定开销
    return total


class LongTermMemory:
    """跨会话长期记忆：JSON 落盘，注入 system prompt，超过上限自动淘汰最旧。"""

    def __init__(self, path: Path, max_entries: int = 50, max_len: int = 500):
        self.path = Path(path)
        self.max_entries = max_entries
        self.max_len = max_len
        self.entries: list[dict] = []  # [{"content", "created"}]
        self.dirty = False             # 有新记忆尚未注入 system prompt
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("entries", [])
            self.entries = [
                {"content": str(e.get("content", ""))[: self.max_len],
                 "created": str(e.get("created", ""))}
                for e in raw if isinstance(e, dict)
            ][: self.max_entries]
        except Exception:
            self.entries = []  # 记忆文件损坏不应影响主程序

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": self.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def save(self, content: str) -> str:
        content = (content or "").strip()[: self.max_len]
        if not content:
            return "错误: 记忆内容为空"
        for e in self.entries:
            if e["content"].lower() == content.lower():
                return f"跳过: 已存在相同记忆: {e['content']}"
        evicted = None
        if len(self.entries) >= self.max_entries:
            evicted = self.entries.pop(0)["content"]
        self.entries.append({
            "content": content,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        self.dirty = True
        self._persist()
        msg = f"OK: 已记住（第 {len(self.entries)} 条）: {content}"
        if evicted:
            msg += f"\n注意: 记忆已满（上限 {self.max_entries} 条），最旧的已被移除: {evicted}"
        return msg

    def list(self) -> str:
        if not self.entries:
            return "(长期记忆为空)"
        return "\n".join(
            f"{i}. {e['content']}  ({e['created'][:10]})"
            for i, e in enumerate(self.entries, 1))

    def forget(self, key: str) -> str:
        """按编号或关键词删除一条记忆。"""
        key = (key or "").strip()
        if not key:
            return "错误: 请给出要删除的编号或关键词"
        target = None
        if key.isdigit() and 1 <= int(key) <= len(self.entries):
            target = self.entries.pop(int(key) - 1)
        if target is None:
            for i, e in enumerate(self.entries):
                if key.lower() in e["content"].lower():
                    target = self.entries.pop(i)
                    break
        if target is None:
            return f"未找到匹配的记忆: {key}"
        self._persist()
        return f"OK: 已删除: {target['content']}"

    def clear(self) -> str:
        n = len(self.entries)
        self.entries = []
        self._persist()
        return f"OK: 已清空 {n} 条长期记忆"

    def render_section(self) -> str:
        """渲染成注入 system prompt 的文本段；无记忆时返回空串。"""
        if not self.entries:
            return ""
        lines = "\n".join(f"- {e['content']}" for e in self.entries)
        return "# 长期记忆（跨会话保存的用户偏好/项目约定/重要事实，供参考）\n" + lines


class ContextCompressor:
    """短期记忆（消息历史）的自动压缩器。

    maybe_compress() 返回压缩后的 token 估算值；未触发压缩时返回 None。
    """

    def __init__(self, llm, budget_tokens: int = 40000, trigger_ratio: float = 0.8,
                 keep_recent: int = 12, on_event: EventCallback | None = None):
        self.llm = llm
        # 不在这里设下限：生产配置的合法性由 config 层保证，测试需要小预算
        self.budget_tokens = max(1, budget_tokens)
        self.trigger_ratio = trigger_ratio
        self.keep_recent = max(2, keep_recent)
        self.on_event = on_event

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _over_budget(self, tokens: int) -> bool:
        return tokens >= self.budget_tokens * self.trigger_ratio

    def maybe_compress(self, messages: list[dict], last_prompt_tokens: int = 0):
        before = max(estimate_tokens(messages), last_prompt_tokens or 0)
        if not self._over_budget(before):
            return None
        tail_len = min(self.keep_recent, len(messages) - 1)
        cut = len(messages) - tail_len
        if cut <= 1:
            return None  # 除 system 外几乎没有历史，无从压起

        # 第一级：机械裁剪（不动消息结构，永不失败）
        self._mechanical_trim(messages, 1, cut)
        after = estimate_tokens(messages)
        if not self._over_budget(after):
            self._emit({"kind": "compress", "before": before, "after": after,
                        "summary": False})
            return after

        # 第二级：LLM 摘要。切分点必须落在 user 消息上（轮次边界），
        # 尾部永远是完整轮次，不会出现 tool_call 悬空。
        boundary = next((i for i in range(cut, len(messages))
                         if messages[i].get("role") == "user"), None)
        if boundary is None or boundary <= 1:
            self._emit({"kind": "compress", "before": before, "after": after,
                        "summary": False})
            return after
        try:
            summary = self._summarize(messages[1:boundary])
        except Exception:
            self._emit({"kind": "compress", "before": before, "after": after,
                        "summary": False})
            return after
        messages[1:boundary] = [{
            "role": "user",
            "content": f"[上下文压缩] 以下是更早对话的摘要，作为背景参考：\n{summary}",
        }]
        after = estimate_tokens(messages)
        self._emit({"kind": "compress", "before": before, "after": after,
                    "summary": True})
        return after

    # ------------------------------------------------------------------
    @staticmethod
    def _mechanical_trim(messages: list[dict], start: int, end: int) -> None:
        for m in messages[start:end]:
            content = m.get("content") or ""
            if m.get("role") == "tool" and len(content) > 400:
                m["content"] = (content[:200]
                                + f"\n[工具输出已压缩，原文 {len(content)} 字符]\n"
                                + content[-100:])
            elif m.get("role") in ("user", "assistant") and len(content) > 800:
                m["content"] = content[:600] + f"\n[消息已压缩，原文 {len(content)} 字符]"

    def _summarize(self, segment: list[dict]) -> str:
        lines = []
        for m in segment:
            role = m.get("role", "?")
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                calls = "; ".join(
                    f"{tc['function']['name']}({str(tc['function']['arguments'])[:120]})"
                    for tc in tool_calls)
                lines.append(f"[assistant 调用工具] {calls}")
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[{role}] {content[:1500]}")
        reply = self.llm.chat([
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": "\n".join(lines)[:60000]},
        ])
        summary = (reply.get("content") or "").strip()
        if not summary:
            raise RuntimeError("摘要为空")
        return summary[:3000]
