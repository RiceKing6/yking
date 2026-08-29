"""文件系统工具：读取、写入、编辑、列目录、搜索。"""
from __future__ import annotations

import locale
import os
import re
from pathlib import Path

from .base import ToolContext

MAX_READ_BYTES = 256 * 1024       # 单次最多读取 256KB
MAX_LIST_ENTRIES = 500
MAX_GREP_MATCHES = 50
MAX_FILE_BYTES_FOR_GREP = 1024 * 1024
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea",
             ".vscode", "dist", "build", ".pytest_cache", ".mypy_cache"}


def resolve_path(ctx: ToolContext, path: str) -> Path:
    """把模型给的路径解析为绝对路径；相对路径基于工作目录。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ctx.workspace / p


def _decode(data: bytes) -> str:
    """依次尝试 utf-8 / 系统编码解码文件内容。"""
    for enc in ("utf-8", locale.getpreferredencoding(False)):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = 2000) -> str:
    """按行读取文件，输出带行号，方便模型在 edit_file 时精确定位。"""
    p = resolve_path(ctx, path)
    if not p.is_file():
        return f"错误: 文件不存在: {p}"
    data = p.read_bytes()[:MAX_READ_BYTES]
    text = _decode(data)
    lines = text.splitlines()
    total = len(lines)
    start = max(1, int(offset))
    if total and start > total:
        return f"错误: offset={start} 超出文件行数（共 {total} 行）"
    end = min(total, start - 1 + max(1, int(limit)))
    out = [f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1)]
    note = ""
    if end < total or start > 1:
        note = f"\n(共 {total} 行，当前显示第 {start}-{end} 行)"
    return "\n".join(out) + note


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """把内容整体写入文件（覆盖已有内容），自动创建父目录。"""
    p = resolve_path(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: 已写入 {len(content.encode('utf-8'))} 字节到 {p}"


def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str,
              replace_all: bool = False) -> str:
    """精确字符串替换：old_string 必须与文件内容完全一致（含缩进与换行）。"""
    p = resolve_path(ctx, path)
    if not p.is_file():
        return f"错误: 文件不存在: {p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        return ("错误: old_string 在文件中不存在。请先用 read_file 查看最新内容，"
                "注意空格、缩进、换行必须完全一致。")
    if count > 1 and not replace_all:
        return (f"错误: old_string 出现了 {count} 次，请加入更多上下文使其唯一，"
                "或设置 replace_all=true 全部替换。")
    if replace_all:
        new_text = text.replace(old_string, new_string)
        replaced = count
    else:
        new_text = text.replace(old_string, new_string, 1)
        replaced = 1
    p.write_text(new_text, encoding="utf-8")
    return f"OK: 已在 {path} 中替换 {replaced} 处"


def list_dir(ctx: ToolContext, path: str = ".") -> str:
    """列出目录下的一层内容（目录在前，文件标注大小）。"""
    p = resolve_path(ctx, path)
    if not p.exists():
        return f"错误: 路径不存在: {p}"
    if p.is_file():
        return f"{p} 是一个文件（{p.stat().st_size} 字节）"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return f"错误: 没有权限读取目录: {p}"
    if not entries:
        return f"{p}/ (空目录)"
    rows = []
    for e in entries[:MAX_LIST_ENTRIES]:
        if e.is_dir():
            rows.append(f"{e.name}/")
        else:
            try:
                rows.append(f"{e.name}  ({e.stat().st_size} B)")
            except OSError:
                rows.append(e.name)
    result = f"{p}/\n" + "\n".join(rows)
    if len(entries) > MAX_LIST_ENTRIES:
        result += f"\n(共 {len(entries)} 项，仅显示前 {MAX_LIST_ENTRIES} 项)"
    return result


def grep_search(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    """在工作目录（或指定子目录/文件）下按正则搜索文本文件，返回 文件:行号:内容。"""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"错误: 无效的正则表达式: {e}"
    root = resolve_path(ctx, path)
    if not root.exists():
        return f"错误: 路径不存在: {root}"

    if root.is_file():
        files = [root]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                files.append(Path(dirpath) / name)

    matches: list[str] = []
    for f in files:
        try:
            if f.stat().st_size > MAX_FILE_BYTES_FOR_GREP:
                continue
            data = f.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:1024]:  # 含 NUL 字节视为二进制文件，跳过
            continue
        text = _decode(data)
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                try:
                    rel = f.relative_to(root)
                except ValueError:
                    rel = f
                matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(matches) >= MAX_GREP_MATCHES:
                    return (f"(已达 {MAX_GREP_MATCHES} 条上限，结果可能不完整)\n"
                            + "\n".join(matches))
    if not matches:
        return f"未找到匹配: pattern={pattern!r} path={root}"
    return "\n".join(matches)
