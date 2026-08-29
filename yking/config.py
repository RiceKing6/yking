"""配置加载：优先级 CLI 参数 > 环境变量(YKING_*) > .env 文件 > 默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录（yking 包的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认对接智谱 GLM（OpenAI 兼容协议）；换服务商只需改 base_url + model
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.6"

# 每个配置项依次尝试读取的环境变量（后面的作为兜底）
_ENV_KEYS = {
    "api_key": ["YKING_API_KEY", "OPENAI_API_KEY"],
    "base_url": ["YKING_BASE_URL"],
    "model": ["YKING_MODEL"],
}


def load_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析：KEY=VALUE，忽略注释与空行；已存在的环境变量不被覆盖。"""
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


@dataclass
class Config:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    workspace: Path = field(default_factory=Path.cwd)
    max_steps: int = 30          # 单轮对话最多走多少步 ReAct 循环
    command_timeout: int = 120   # run_command 超时（秒）
    auto_approve: bool = False   # True 则 run_command 不询问直接执行
    stream: bool = True          # 流式输出（边生成边显示）
    parallel: int = 1            # Plan 模式并行执行的无依赖任务数（1 = 串行）
    memory: bool = True          # 长期记忆（跨会话，存于工作目录 .yking/memory.json）
    context_budget: int = 40000  # 短期上下文 token 预算（超过自动压缩）


def load_config(args) -> Config:
    """从 CLI 参数、环境变量、.env 文件组装配置。args 为 argparse 解析结果。"""
    # 依次加载当前目录与项目根目录的 .env
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(PROJECT_ROOT / ".env")

    def pick(name: str, cli_value) -> str:
        if cli_value:
            return str(cli_value)
        for env_name in _ENV_KEYS[name]:
            value = os.environ.get(env_name)
            if value:
                return value
        return ""

    cfg = Config(
        api_key=pick("api_key", args.api_key),
        base_url=(pick("base_url", args.base_url) or DEFAULT_BASE_URL).rstrip("/"),
        model=pick("model", args.model) or DEFAULT_MODEL,
    )
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.workspace:
        cfg.workspace = Path(args.workspace).expanduser().resolve()
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    cfg.auto_approve = bool(args.auto_approve) or os.environ.get(
        "YKING_AUTO_APPROVE", ""
    ).strip().lower() in ("1", "true", "yes")
    cfg.stream = (
        not bool(args.no_stream)
        and os.environ.get("YKING_STREAM", "1").strip().lower() not in ("0", "false", "no")
    )
    if getattr(args, "parallel", None):
        cfg.parallel = max(1, int(args.parallel))
    cfg.memory = (
        not bool(getattr(args, "no_memory", False))
        and os.environ.get("YKING_MEMORY", "1").strip().lower() not in ("0", "false", "no")
    )
    budget = getattr(args, "context_budget", None) or os.environ.get("YKING_CONTEXT_BUDGET", "")
    if budget:
        try:
            cfg.context_budget = max(4000, int(budget))
        except ValueError:
            pass
    return cfg
