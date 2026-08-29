"""Multi-Agent（Orchestrator + Planner/Worker/Reviewer）冒烟测试（无需 API Key）。

覆盖：规划解析（type/acceptance）、正常两步流程（Worker 执行 + Reviewer 工具化核验）、
验收打回 → 带问题返工 → 通过、连续打回耗尽 → 步骤失败 + 依赖级联跳过、
Worker 自报失败不走 Reviewer、Reviewer 故障时保守放行（fail-open）。

运行: conda run -n yking python tests/test_multi_agent.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yking.config import Config                                    # noqa: E402
from yking.memory import LongTermMemory                            # noqa: E402
from yking.orchestrator import (READONLY_TOOLS, Orchestrator,       # noqa: E402
                                filter_registry)
from yking.plan import PlanError, parse_plan                       # noqa: E402
from yking.tools import ToolContext, build_registry                # noqa: E402


def submit_plan_call(args: dict, call_id: str = "p1") -> dict:
    return {"content": "", "usage": {},
            "tool_calls": [{"id": call_id, "name": "submit_plan",
                            "arguments": json.dumps(args, ensure_ascii=False)}]}


def review_call(args: dict, call_id: str) -> dict:
    return {"content": "", "usage": {},
            "tool_calls": [{"id": call_id, "name": "submit_review",
                            "arguments": json.dumps(args, ensure_ascii=False)}]}


def tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"content": "", "usage": {},
            "tool_calls": [{"id": call_id, "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False)}]}


def final(text: str) -> dict:
    return {"content": text, "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class RoleFakeLLM:
    """按请求里的工具集路由到各角色的脚本队列：
    submit_plan → planner；submit_review → reviewer；其余按「步骤 [N]」→ worker:N。"""

    def __init__(self, scripts: dict):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls = []  # (key, messages)

    def chat(self, messages, tools=None, on_text_delta=None):
        names = {t["function"]["name"] for t in (tools or [])}
        blob = "\n".join((m.get("content") or "") for m in messages)
        if "submit_plan" in names:
            key = "planner"
        elif "submit_review" in names:
            key = "reviewer"
        else:
            m = re.search(r"步骤 \[(\w+)\]", blob)
            key = f"worker:{m.group(1)}" if m else "worker:?"
        self.calls.append((key, list(messages)))
        script = self.scripts.get(key)
        assert script, f"没有 {key} 的脚本"
        return script.pop(0)


def make_orchestrator(root: Path, scripts: dict, with_memory=False):
    memory = LongTermMemory(root / ".yking" / "memory.json") if with_memory else None
    ctx = ToolContext(workspace=root, auto_approve=True, memory=memory)
    full = build_registry(ctx)
    cfg = Config(api_key="dummy", workspace=root)
    llm = RoleFakeLLM(scripts)
    events = []
    orch = Orchestrator(
        llm=llm,
        worker_registry=full,
        reviewer_registry=filter_registry(full, READONLY_TOOLS),
        ctx=ctx, cfg=cfg, memory=memory, on_event=events.append,
    )
    return orch, llm, events


PLAN_ARGS = {"goal": "演示", "tasks": [
    {"id": "1", "title": "创建文件", "type": "code",
     "goal": "创建 a.txt，内容为 X", "acceptance": "a.txt 存在且内容为 X", "deps": []},
    {"id": "2", "title": "命令验证", "type": "command",
     "goal": "运行命令查看 a.txt 内容", "acceptance": "命令输出包含 X", "deps": ["1"]},
]}


def test_parse_plan_typed() -> None:
    plan = parse_plan({"goal": "g", "tasks": [
        {"id": "1", "title": "t", "goal": "g1", "deps": [], "type": "research",
         "acceptance": "结论覆盖 A 和 B"},
        {"id": "2", "title": "t2", "goal": "g2", "deps": ["1"]},
    ]})
    assert plan.task("1").type == "research" and plan.task("1").acceptance != ""
    assert plan.task("2").type == "code" and plan.task("2").acceptance == ""
    try:
        parse_plan({"goal": "g", "tasks": [
            {"id": "1", "title": "t", "goal": "g", "deps": [], "type": "hack"}]})
        raise AssertionError("非法 type 应当报错")
    except PlanError:
        pass
    print("  [ok] 计划解析：type 归一化校验 + acceptance 字段")


def test_happy_path(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call(PLAN_ARGS)],
        "worker:1": [tool_call("w1", "write_file", {"path": "a.txt", "content": "X"}),
                     final("已创建 a.txt，内容为 X，与验收标准一致。")],
        "reviewer": [
            tool_call("r0", "read_file", {"path": "a.txt"}),   # Reviewer 实际核验
            review_call({"verdict": "pass", "summary": "a.txt 存在且内容为 X", "issues": []},
                        call_id="r1"),
            review_call({"verdict": "pass", "summary": "命令输出包含 X", "issues": []},
                        call_id="r2"),
        ],
        "worker:2": [tool_call("w2", "run_command", {"command": "type a.txt"}),
                     final("命令输出 X，符合验收标准。")],
    }
    orch, llm, events = make_orchestrator(root, scripts)
    plan = orch.execute(orch.plan("演示流程"))

    assert plan.task("1").status == "done" and plan.task("2").status == "done"
    assert (root / "a.txt").read_text(encoding="utf-8") == "X"
    assert "Reviewer 结论" in plan.task("1").report
    keys = [k for k, _ in llm.calls]
    assert keys == ["planner",
                    "worker:1", "worker:1",       # 工具调用 + 执行报告
                    "reviewer", "reviewer",       # 读文件核验 + 提交结论
                    "worker:2", "worker:2",
                    "reviewer"], keys
    reviews = [e for e in events if e["kind"] == "review"]
    assert len(reviews) == 2 and all(r["verdict"] == "pass" for r in reviews)
    # Reviewer 的核验工具调用带角色标签
    assert any(e.get("role") == "reviewer" and e["kind"] == "tool_call" for e in events)
    assert events[-1]["kind"] == "plan_finished"
    print("  [ok] 正常流程：Planner → Worker 执行 → Reviewer 读文件核验 → pass，依赖顺序正确")


def test_review_fail_then_retry(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "创建文件", "type": "code",
             "goal": "创建 a.txt，内容为 X", "acceptance": "a.txt 内容为 X", "deps": []}]})],
        "worker:1": [
            tool_call("w1", "write_file", {"path": "a.txt", "content": "Y"}),
            final("已写入 a.txt。"),
            tool_call("w2", "write_file", {"path": "a.txt", "content": "X"}),
            final("已修正：a.txt 现在内容为 X。"),
        ],
        "reviewer": [
            review_call({"verdict": "fail", "summary": "内容不符",
                         "issues": ["a.txt 内容是 Y，验收要求 X"]}, call_id="r1"),
            review_call({"verdict": "pass", "summary": "a.txt 内容已是 X", "issues": []},
                        call_id="r2"),
        ],
    }
    orch, llm, events = make_orchestrator(root, scripts)
    plan = orch.execute(orch.plan("打回重做演示"))

    assert plan.task("1").status == "done"
    assert (root / "a.txt").read_text(encoding="utf-8") == "X"
    worker_call_msgs = [m for k, m in llm.calls if k == "worker:1"]
    assert len(worker_call_msgs) == 4, "两轮尝试 × 每轮 2 次调用（工具 + 报告）"
    kinds = [(e["kind"], e.get("verdict")) for e in events]
    assert ("review", "fail") in kinds and ("review", "pass") in kinds
    # 返工提示里带上 Reviewer 的问题清单
    retry = next(e for e in events if e["kind"] == "worker_retry")
    assert "a.txt 内容是 Y" in retry["reason"]
    # 第二轮第一次调用的指令里应包含 Reviewer 的问题清单
    assert any("a.txt 内容是 Y" in (m.get("content") or "") for m in worker_call_msgs[2])
    print("  [ok] 验收打回：fail + 问题清单 → Worker 带问题返工 → 二轮通过")


def test_exhausted_attempts(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "总失败", "type": "code",
             "goal": "g1", "acceptance": "a1", "deps": []},
            {"id": "2", "title": "依赖步骤", "type": "code",
             "goal": "g2", "acceptance": "a2", "deps": ["1"]}]}),
        ],
        "worker:1": [final("尝试报告") for _ in range(3)],
        "reviewer": [
            review_call({"verdict": "fail", "summary": "不达标", "issues": ["问题 A"]}, "r1"),
            review_call({"verdict": "fail", "summary": "仍不达标", "issues": ["问题 B"]}, "r2"),
            review_call({"verdict": "fail", "summary": "依旧不达标", "issues": ["问题 C"]}, "r3"),
        ],
    }
    orch, llm, events = make_orchestrator(root, scripts)
    plan = orch.execute(orch.plan("耗尽演示"))

    assert plan.task("1").status == "failed"
    assert plan.task("2").status == "skipped", "依赖失败步骤应级联跳过"
    assert "问题 C" in plan.task("1").report
    fail_reviews = [e for e in events if e["kind"] == "review" and e["verdict"] == "fail"]
    assert len(fail_reviews) == 3
    kinds = [e["kind"] for e in events]
    assert "task_failed" in kinds and "task_skipped" in kinds
    assert kinds[-1] == "plan_finished"
    print("  [ok] 连续打回：3 次未通过 → 步骤失败，依赖步骤级联跳过")


def test_worker_self_failed(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "自报失败", "type": "code",
             "goal": "g1", "acceptance": "a1", "deps": []}]}),
        ],
        "worker:1": [
            final("[FAILED]: 缺少依赖库 xyz"),
            tool_call("w1", "write_file", {"path": "ok.txt", "content": "done"}),
            final("第二次尝试成功。"),
        ],
        "reviewer": [review_call({"verdict": "pass", "summary": "ok.txt 存在", "issues": []},
                                 call_id="r1")],
    }
    orch, llm, events = make_orchestrator(root, scripts)
    plan = orch.execute(orch.plan("自报失败演示"))

    assert plan.task("1").status == "done"
    reviewer_calls = [k for k, _ in llm.calls if k == "reviewer"]
    assert len(reviewer_calls) == 1, "Worker 自报失败的那轮不应打扰 Reviewer"
    retry = next(e for e in events if e["kind"] == "worker_retry")
    assert "缺少依赖库" in retry["reason"]
    print("  [ok] Worker 自报失败：跳过验收直接带原因返工")


def test_reviewer_fail_open(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "t", "type": "code",
             "goal": "g1", "acceptance": "a1", "deps": []}]}),
        ],
        "worker:1": [final("完成了")],
        "reviewer": [],  # Reviewer 脚本耗尽 → 每次调用都抛错
    }
    orch, llm, events = make_orchestrator(root, scripts)
    plan = orch.execute(orch.plan("fail-open 演示"))

    assert plan.task("1").status == "done"
    assert "未经审查" in plan.task("1").report
    assert not [e for e in events if e["kind"] == "review"], "不应有验收结论"
    print("  [ok] Reviewer 故障：保守放行并在报告中标注未经审查（fail-open）")


def test_memory_in_worker_prompt(root: Path) -> None:
    scripts = {
        "planner": [submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "t", "type": "code",
             "goal": "g1", "acceptance": "a1", "deps": []}]}),
        ],
        "worker:1": [final("完成")],
        "reviewer": [review_call({"verdict": "pass", "summary": "ok", "issues": []}, "r1")],
    }
    orch, llm, events = make_orchestrator(root, scripts, with_memory=True)
    orch.memory.save("这个项目统一使用 pytest")
    plan = orch.execute(orch.plan("记忆注入演示"))
    assert plan.task("1").status == "done"
    worker_system = llm.calls[1][1][0]["content"]  # 第一次 Worker 调用的 system
    assert "pytest" in worker_system, "长期记忆应注入 Worker 的 system prompt"
    print("  [ok] 长期记忆注入 Worker 角色的 system prompt")


def main() -> int:
    print("yking Multi-Agent 冒烟测试:")
    test_parse_plan_typed()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_happy_path(root)
        test_review_fail_then_retry(root)
        test_exhausted_attempts(root)
        test_worker_self_failed(root)
        test_reviewer_fail_open(root)
        test_memory_in_worker_prompt(root)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
