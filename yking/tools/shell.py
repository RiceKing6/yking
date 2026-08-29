"""命令执行工具。"""
from __future__ import annotations

import locale
import subprocess

from .base import ToolContext
from .fs import resolve_path

MAX_OUTPUT_CHARS = 12000


def _decode(data: bytes) -> str:
    """依次尝试 utf-8 / gbk / 系统编码解码命令输出（Windows 中文环境常用 gbk）。"""
    if not data:
        return ""
    for enc in ("utf-8", "gbk", locale.getpreferredencoding(False)):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head, tail = 4000, 6000
    return (text[:head]
            + f"\n...(输出过长，中间省略 {len(text) - head - tail} 字符)...\n"
            + text[-tail:])


def run_command(ctx: ToolContext, command: str, cwd: str = ".") -> str:
    """在 shell 中执行一条命令，返回 exit_code / stdout / stderr。"""
    workdir = resolve_path(ctx, cwd)
    if not workdir.is_dir():
        return f"错误: 工作目录不存在: {workdir}"
    if ctx.confirm and not ctx.auto_approve:
        if not ctx.confirm(command):
            return "用户拒绝执行该命令。请改用其他方式，或向用户解释你想做什么。"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            timeout=ctx.command_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"错误: 命令执行超时（>{ctx.command_timeout} 秒）: {command}"
    except OSError as e:
        return f"错误: 命令启动失败: {e}"
    out = (
        f"exit_code={proc.returncode}\n"
        f"--- stdout ---\n{_decode(proc.stdout) or '(空)'}\n"
        f"--- stderr ---\n{_decode(proc.stderr) or '(空)'}"
    )
    return _truncate(out)
