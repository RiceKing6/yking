"""工具基础定义：ToolContext（共享上下文）与 Tool（描述 + 实现）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # 仅类型标注用，避免运行时依赖
    from ..memory import LongTermMemory


@dataclass
class ToolContext:
    """所有工具共享的运行上下文。"""
    workspace: Path
    command_timeout: int = 120
    auto_approve: bool = False
    # run_command 执行前的确认回调；返回 False 表示用户拒绝
    confirm: Callable[[str], bool] | None = None
    # 长期记忆（可选）：启用后注册 save_memory 工具、system prompt 注入记忆
    memory: "LongTermMemory | None" = None


@dataclass
class Tool:
    """一个可被模型调用的工具：JSON Schema 描述 + Python 实现。"""
    name: str
    description: str
    parameters: dict             # JSON Schema，描述参数
    func: Callable[..., str]     # 统一签名 func(ctx, **kwargs) -> str（观察结果）

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
