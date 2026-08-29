"""并行执行 DAG 无依赖任务的冒烟测试（无需 API Key）。

覆盖：并行时真实并发峰值 >= 2、串行时并发恒为 1、依赖顺序不被打乱、
并行模式下一个任务失败时等其他任务收尾再重规划。

运行: conda run -n yking python tests/test_parallel.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yking.config import Config                      # noqa: E402
from yking.planner import Planner, PlanExecutor      # noqa: E402
from yking.tools import ToolContext, build_registry  # noqa: E402


def submit_plan_call(args: dict, call_id: str = "p1") -> dict:
    return {"content": "", "usage": {},
            "tool_calls": [{"id": call_id, "name": "submit_plan",
                            "arguments": json.dumps(args, ensure_ascii=False)}]}


class FakeLLM:
    """线程安全的脚本化 LLM。

    根据消息里的「请完成任务 [N]」标记把脚本按任务分发；
    统计并发调用峰值（max_active）来验证是否真的并行。
    """

    def __init__(self, scripts: dict, work: float = 0.15):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.work = work
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []

    def chat(self, messages, tools, on_text_delta=None):
        blob = "\n".join((m.get("content") or "") for m in messages)
        m = re.search(r"请完成任务 \[(\w+)\]", blob)
        key = m.group(1) if m else "plan"
        with self.lock:
            self.calls.append((key, list(messages)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            script = self.scripts.get(key)
            assert script, f"没有 key={key} 的脚本"
            reply = script.pop(0)
        time.sleep(self.work)  # 模拟推理耗时，制造并发窗口
        with self.lock:
            self.active -= 1
        return reply


def make_plan_args():
    return {"goal": "并行演示", "tasks": [
        {"id": "1", "title": "任务一", "goal": "g1", "deps": []},
        {"id": "2", "title": "任务二", "goal": "g2", "deps": []},
        {"id": "3", "title": "汇总任务", "goal": "g3", "deps": ["1", "2"]},
    ]}


def run_plan(root: Path, parallel: int, scripts: dict) -> tuple:
    ctx = ToolContext(workspace=root, auto_approve=True)
    registry = build_registry(ctx)
    cfg = Config(api_key="dummy", workspace=root, parallel=parallel)
    llm = FakeLLM(scripts)
    events = []
    planner = Planner(llm, on_event=events.append)
    plan = planner.create_plan("并行演示")
    executor = PlanExecutor(llm, registry, ctx, cfg, on_event=events.append)
    plan = executor.run(plan)
    return plan, events, llm


def test_parallel_runs_independent_tasks(root: Path) -> None:
    plan, events, llm = run_plan(root, parallel=2, scripts={
        "plan": [submit_plan_call(make_plan_args())],
        "1": [{"content": "任务1完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        "2": [{"content": "任务2完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        "3": [{"content": "任务3完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
    })

    assert llm.max_active >= 2, f"两个无依赖任务应并行执行，实测并发峰值 {llm.max_active}"
    assert all(t.status == "done" for t in plan.tasks), plan.tasks

    # 依赖顺序不被打乱：汇总任务必须在 1、2 都完成之后才开始
    seq = [(e["kind"], e.get("task")) for e in events]
    start3 = seq.index(("task_start", "3"))
    assert seq.index(("task_done", "1")) < start3
    assert seq.index(("task_done", "2")) < start3
    assert seq[-1] == ("plan_finished", None), seq[-5:]
    print(f"  [ok] 并行执行：无依赖任务并发峰值 = {llm.max_active}，依赖顺序正确")


def test_sequential_stays_sequential(root: Path) -> None:
    plan, events, llm = run_plan(root, parallel=1, scripts={
        "plan": [submit_plan_call(make_plan_args())],
        "1": [{"content": "任务1完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        "2": [{"content": "任务2完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        "3": [{"content": "任务3完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
    })

    assert llm.max_active == 1, f"串行模式并发应恒为 1，实测 {llm.max_active}"
    assert all(t.status == "done" for t in plan.tasks)
    seq = [(e["kind"], e.get("task")) for e in events]
    assert seq.index(("task_start", "2")) > seq.index(("task_done", "1"))
    print("  [ok] 串行模式（parallel=1）：一次只跑一个任务")


def test_parallel_failure_waits_and_replans(root: Path) -> None:
    """一个任务失败时，另一个在跑的任务会被等完成后，才带着完整进度重规划。"""
    plan, events, llm = run_plan(root, parallel=2, scripts={
        "plan": [
            submit_plan_call({"goal": "并行演示", "tasks": [
                {"id": "1", "title": "会失败", "goal": "g1", "deps": []},
                {"id": "2", "title": "会成功", "goal": "g2", "deps": []},
            ]}),
            # 重规划：放弃
            submit_plan_call({"goal": "并行演示", "tasks": [],
                              "abort_reason": "无法继续"}, call_id="p2"),
        ],
        "1": [{"content": "[FAILED]: 故意失败", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        "2": [{"content": "任务2完成。", "tool_calls": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
    })

    assert plan.task("1").status == "failed"
    assert plan.task("2").status == "done"   # 在跑的兄弟任务被等完并保留成果
    kinds = [e["kind"] for e in events]
    assert kinds.count("replanning") == 1
    assert kinds[-1] == "plan_aborted"
    # 重规划器看到的是包含任务2成果的完整进度
    replan_blob = "\n".join((m.get("content") or "")
                            for key, msgs in llm.calls if key == "plan"
                            for m in msgs)
    assert "[done] 2" in replan_blob, "重规划输入应包含任务2已完成的信息"
    print("  [ok] 并行失败处理：等在跑任务收尾后再重规划，进度完整")


def main() -> int:
    print("yking 并行执行冒烟测试:")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_parallel_runs_independent_tasks(root)
        test_sequential_stays_sequential(root)
        test_parallel_failure_waits_and_replans(root)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
