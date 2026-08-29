"""工具注册表：在这里把名字、Schema 与实现函数组装成 Tool。

新增工具的步骤：
  1. 在 yking/tools/ 下写函数，签名统一为 (ctx: ToolContext, **参数) -> str，
     返回的字符串就是模型看到的"观察结果"；
  2. 在下面的 build_registry 里注册名字、描述和 JSON Schema。
"""
from __future__ import annotations

from .base import Tool, ToolContext
from . import fs, shell

__all__ = ["Tool", "ToolContext", "build_registry", "execute_tool"]


def _save_memory(ctx: ToolContext, content: str) -> str:
    if ctx.memory is None:
        return "错误: 长期记忆未启用"
    return ctx.memory.save(content)


def build_registry(ctx: ToolContext) -> dict[str, Tool]:
    tools = [
        Tool(
            name="read_file",
            description="读取文本文件内容，输出带行号。大文件可用 offset/limit 分段读取。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，相对工作目录或绝对路径"},
                    "offset": {"type": "integer", "description": "起始行号（从 1 开始），默认 1"},
                    "limit": {"type": "integer", "description": "最多读取的行数，默认 2000"},
                },
                "required": ["path"],
            },
            func=fs.read_file,
        ),
        Tool(
            name="write_file",
            description="把内容整体写入文件（覆盖已有内容），自动创建父目录。适合新建文件或整体重写。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "完整的文件内容"},
                },
                "required": ["path", "content"],
            },
            func=fs.write_file,
        ),
        Tool(
            name="edit_file",
            description="对已有文件做精确字符串替换。old_string 必须与文件中的内容完全一致（含缩进），且默认必须唯一。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "old_string": {"type": "string",
                                   "description": "要被替换的原文，必须与文件内容完全一致"},
                    "new_string": {"type": "string", "description": "替换后的新内容"},
                    "replace_all": {"type": "boolean",
                                    "description": "old_string 出现多次时是否全部替换，默认 false"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            func=fs.edit_file,
        ),
        Tool(
            name="list_dir",
            description="列出目录下的一层内容（目录在前，文件标注大小）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认工作目录根"},
                },
            },
            func=fs.list_dir,
        ),
        Tool(
            name="grep_search",
            description="按正则表达式搜索文件内容，返回 文件:行号:内容。适合全局查找代码。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式"},
                    "path": {"type": "string", "description": "搜索的根目录或单个文件，默认整个工作目录"},
                },
                "required": ["pattern"],
            },
            func=fs.grep_search,
        ),
        Tool(
            name="run_command",
            description="在 shell 中执行一条命令（运行测试、脚本、pip、git 等），返回 exit_code、stdout、stderr。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "cwd": {"type": "string", "description": "执行时所在的目录，默认工作目录根"},
                },
                "required": ["command"],
            },
            func=shell.run_command,
        ),
    ]
    if ctx.memory is not None:
        tools.append(Tool(
            name="save_memory",
            description=("把值得跨会话记住的信息写入长期记忆（用户偏好、项目约定、重要决定）。"
                         "仅在用户明确表达偏好/要求，或出现稳定的项目约定时使用；"
                         "不要保存一次性任务细节。"),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的一句话事实"},
                },
                "required": ["content"],
            },
            func=_save_memory,
        ))
    return {t.name: t for t in tools}


def execute_tool(registry: dict[str, Tool], ctx: ToolContext,
                 name: str, arguments: dict) -> str:
    """执行一个工具调用。任何异常都转成错误字符串返回给模型，让它有机会自我修正。"""
    tool = registry.get(name)
    if tool is None:
        return f"错误: 不存在名为 {name!r} 的工具。可用工具: {', '.join(registry)}"
    try:
        return str(tool.func(ctx, **arguments))
    except TypeError as e:
        return f"错误: 工具参数不合法: {e}"
    except Exception as e:
        return f"错误: {type(e).__name__}: {e}"
