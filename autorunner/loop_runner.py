"""loop_runner: 自动循环实验主循环"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 支持直接运行: python autorunner/loop_runner.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.claude_code_agent import ClaudeCodeAgent
from autorunner.experiment_executor import ExperimentResult, run as run_experiment
from autorunner.failure_analyzer import analyze as analyze_failure
from autorunner.metric_judge import judge as metric_judge


WORKDIR = Path("/mount/disk1/rl-hyr/autoresearch")
ARTIFACTS_DIR = WORKDIR / "autorunner" / "artifacts"
MODEL = "MiniMax-M2.7"


def _load_baseline() -> float | None:
    """从 results.tsv 加载当前最佳 val_bpb（仅 status=keep 的行）"""
    results_tsv = WORKDIR / "results.tsv"
    if not results_tsv.exists():
        return None

    best = None
    for line in results_tsv.read_text().splitlines():
        if line.startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        status = parts[3].strip()
        if status != "keep":
            continue
        try:
            bpb = float(parts[1])
            if best is None or bpb < best:
                best = bpb
        except ValueError:
            continue
    return best


def _extract_code(content: str) -> str:
    """从 LLM 输出中提取 train.py 代码（去除 markdown fences）"""
    lines = content.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            start = i + 1
            break
    else:
        start = 0

    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "```":
            end = i
            break

    code = "\n".join(lines[start:end])
    return code.strip()


def _git_commit(decision: str, description: str = "") -> str:
    """Git commit 当前 train.py，返回 commit hash"""
    try:
        subprocess.run(["git", "add", "train.py"], cwd=WORKDIR, check=True)
        msg = f"{decision}: {description}" if description else decision
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _write_llm_log(iteration: int, phase: str, content: str):
    """LLM 回复写入日志文件"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    llm_log = ARTIFACTS_DIR / f"llm_iter-{iteration:03d}_{phase}.log"
    llm_log.write_text(content)


def _print_progress(
    iteration: int, max_iter: int, label: str, status: str, extra: str = ""
):
    """终端打印一行进度"""
    symbols = {"start": "⏳", "done": "🗸", "fail": "🗴", "skip": "»"}
    sym = symbols.get(status, "  ")
    print(f"[{iteration}/{max_iter}] {label} {sym} {extra}")


def _build_run_summaries() -> list[str]:
    """从 results.tsv 构建所有历史实验的摘要"""
    results_tsv = WORKDIR / "results.tsv"
    if not results_tsv.exists():
        return []

    summaries = []
    for line in results_tsv.read_text().splitlines():
        if line.startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        commit = parts[0]
        val_bpb = parts[1]
        status = parts[3].strip()
        description = parts[4].strip()
        summaries.append(
            f"commit={commit} val_bpb={val_bpb} status={status} desc={description}"
        )
    return summaries


def _write_iter_artifacts(
    iteration: int,
    exp_result: ExperimentResult,
    improved: bool,
    decision: str,
    git_commit_hash: str,
):
    """写入单轮产物 iter-<n>.json"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "iteration": iteration,
        "git_commit": git_commit_hash,
        "candidate_id": f"cand_{iteration:03d}",
        "status": exp_result.status,
        "metrics": {
            "val_bpb": exp_result.val_bpb,
            "peak_vram_mb": exp_result.peak_vram_mb,
            "training_seconds": exp_result.training_seconds,
            "mfu_percent": exp_result.mfu_percent,
            "total_tokens_M": exp_result.total_tokens_M,
            "num_steps": exp_result.num_steps,
        },
        "primary_metric": exp_result.val_bpb,
        "direction": "minimize",
        "improved": improved,
        "decision": decision,
        "failure_type": exp_result.failure_type,
        "next_action": (
            "continue"
            if improved
            else "stop_early" if decision == "discard" else "retry_with_fix"
        ),
        "notes": f"status={exp_result.status}, failure={exp_result.failure_type}",
    }
    iter_file = ARTIFACTS_DIR / f"iter-{iteration:03d}.json"
    iter_file.write_text(json.dumps(entry, indent=2))


def _append_history(
    iteration: int,
    exp_result: ExperimentResult,
    improved: bool,
    decision: str,
    git_commit_hash: str,
):
    """追加到 history.jsonl"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history_file = ARTIFACTS_DIR / "history.jsonl"
    entry = {
        "iteration": iteration,
        "git_commit": git_commit_hash,
        "primary_metric": exp_result.val_bpb,
        "improved": improved,
        "decision": decision,
        "failure_type": exp_result.failure_type,
        "notes": f"status={exp_result.status}",
    }
    with history_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _update_best_candidate(
    iteration: int,
    exp_result: ExperimentResult,
    git_commit_hash: str,
):
    """更新 best_candidate.json"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best_file = ARTIFACTS_DIR / "best_candidate.json"
    if exp_result.val_bpb is None:
        return
    current = None
    if best_file.exists():
        try:
            current = json.loads(best_file.read_text())
        except json.JSONDecodeError:
            current = None
    if current is None or exp_result.val_bpb < current.get(
        "primary_metric", float("inf")
    ):
        entry = {
            "iteration": iteration,
            "git_commit": git_commit_hash,
            "primary_metric": exp_result.val_bpb,
            "peak_vram_mb": exp_result.peak_vram_mb,
            "mfu_percent": exp_result.mfu_percent,
            "num_steps": exp_result.num_steps,
            "timestamp": datetime.now().isoformat(),
        }
        best_file.write_text(json.dumps(entry, indent=2))


def _write_run_summary(stop_reason: str, best_bpb: float):
    """写入 run_summary.json"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = ARTIFACTS_DIR / "run_summary.json"
    entry = {
        "stop_reason": stop_reason,
        "best_bpb": best_bpb,
        "timestamp": datetime.now().isoformat(),
    }
    summary_file.write_text(json.dumps(entry, indent=2))


def _sync_history_from_results_tsv():
    """把 results.tsv 中的历史记录同步到 history.jsonl（按 commit 去重）"""
    results_tsv = WORKDIR / "results.tsv"
    history_file = ARTIFACTS_DIR / "history.jsonl"
    if not results_tsv.exists():
        return

    # 已有 history.jsonl 中的 commit 集合
    existing_commits = set()
    if history_file.exists():
        for line in history_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                existing_commits.add(entry.get("git_commit", ""))
            except json.JSONDecodeError:
                continue

    # 逐行读 results.tsv，追加不在 existing_commits 中的记录
    history_lines = []
    for line in results_tsv.read_text().splitlines():
        if line.startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        commit = parts[0].strip()
        if commit in existing_commits:
            continue
        try:
            val_bpb = float(parts[1])
        except ValueError:
            val_bpb = None
        status = parts[3].strip()
        description = parts[4].strip()
        entry = {
            "git_commit": commit,
            "primary_metric": val_bpb,
            "decision": "keep" if status == "keep" else "discard",
            "notes": f"from results.tsv: {description}",
        }
        history_lines.append(json.dumps(entry))

    if history_lines:
        with history_file.open("a") as f:
            for line in history_lines:
                f.write(line + "\n")


def _append_results_tsv(
    git_commit_hash: str,
    exp_result: ExperimentResult,
    decision: str,
    iteration: int,
    description: str,
):
    """追加一行到 results.tsv"""
    results_tsv = WORKDIR / "results.tsv"
    val_bpb = exp_result.val_bpb if exp_result.val_bpb is not None else 0.0
    peak_vram = exp_result.peak_vram_mb if exp_result.peak_vram_mb is not None else 0.0
    status = "keep" if decision == "keep" else "discard"
    line = f"{git_commit_hash}\t{val_bpb:.6f}\t{peak_vram:.1f}\t{status}\t{description}\n"
    with results_tsv.open("a") as f:
        f.write(line)


def _extract_description_from_llm_reply(reply: str, iteration: int, prefix: str = "") -> str:
    """从 LLM 回复中提取 DESCRIPTION 字段，失败时回退到首行文本。"""
    fallback = f"iter {iteration}"
    if not reply:
        return f"{prefix}{fallback}" if prefix else fallback

    picked = ""
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("DESCRIPTION:"):
            picked = line.split(":", 1)[1].strip()
            break

    if not picked:
        for raw in reply.splitlines():
            line = raw.strip()
            if not line or line.startswith("```"):
                continue
            picked = line.lstrip("-•* ")
            if picked:
                break

    if not picked:
        picked = fallback

    # 清理 TSV 敏感字符，避免破坏列结构
    picked = picked.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    picked = " ".join(picked.split())
    if prefix:
        picked = f"{prefix}{picked}"
    return picked[:160]


def _repair_once_on_failure(
    *,
    agent: ClaudeCodeAgent,
    iteration: int,
    max_iterations: int,
    exp_result: ExperimentResult,
    time_budget: int,
) -> tuple[ExperimentResult, bool, str | None, str | None]:
    """失败时进行一次修复并重跑，返回（实验结果，是否已修复，修复提交哈希）。"""
    # 仅对可修复失败（依赖/运行时）触发重试，避免 timeout/数值异常死循环
    analysis = analyze_failure(exp_result.failure_type or "runtime", exp_result.log_tail)
    if not analysis.should_retry:
        return exp_result, False, None, None

    _print_progress(iteration, max_iterations, "LLM (repair)", "start")
    t0 = time.monotonic()

    current_code = (WORKDIR / "train.py").read_text()
    issues = (
        f"failure_type: {analysis.failure_type}\n"
        f"repair_strategy: {analysis.repair_strategy}\n\n"
        "错误日志尾部（最多 50 行）：\n"
        f"{exp_result.log_tail}"
    )
    repair_result = agent.repair(
        files={"train.py": current_code},
        issues=issues,
        workdir=WORKDIR,
    )

    elapsed = time.monotonic() - t0
    _write_llm_log(iteration, "repair", repair_result.content)

    if not repair_result.success:
        _print_progress(
            iteration,
            max_iterations,
            "LLM (repair)",
            "fail",
            f"({elapsed:.1f}s) rc={repair_result.rc}",
        )
        return exp_result, False, None, None

    # 仅接受 Claude 在工作区原地修改后的 train.py
    repaired_code = repair_result.files.get("train.py", "").strip()
    if not repaired_code:
        _print_progress(
            iteration,
            max_iterations,
            "LLM (repair)",
            "fail",
            "(train.py not updated)",
        )
        return exp_result, False, None, None

    if repaired_code == current_code:
        _print_progress(
            iteration,
            max_iterations,
            "LLM (repair)",
            "skip",
            "(no code change)",
        )
        return exp_result, False, None, None

    (WORKDIR / "train.py").write_text(repaired_code)
    repair_git_hash = _git_commit("repair", f"iteration {iteration}")
    _print_progress(iteration, max_iterations, "LLM (repair)", "done", f"({elapsed:.1f}s)")

    # 修复后只重跑一次，若仍失败则放弃本轮
    _print_progress(iteration, max_iterations, "Run experiment (after repair)", "start")
    rerun_result = run_experiment(
        train_py_path=WORKDIR / "train.py",
        time_budget=time_budget,
    )
    runtime_min = rerun_result.runtime_seconds / 60
    _print_progress(
        iteration,
        max_iterations,
        "Run experiment (after repair)",
        "done",
        f"({runtime_min:.1f}m)",
    )

    repair_desc = _extract_description_from_llm_reply(
        repair_result.content,
        iteration=iteration,
        prefix="repair: ",
    )
    return rerun_result, True, repair_git_hash or None, repair_desc


def run_loop(
    max_iterations: int = 10,
    early_stop: int = 3,
    time_budget: int = 600,
    topic: str | None = None,
):
    """
    主循环

    Args:
        max_iterations: 最大迭代次数
        early_stop: 连续多少轮无提升后停止
        time_budget: 每次训练的时间预算（秒）
        topic: 实验主题描述
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_bpb = _load_baseline()
    if baseline_bpb is None:
        baseline_bpb = float("inf")
        print(f"[loop_runner] No baseline found, starting from scratch")
    else:
        print(f"[loop_runner] Baseline val_bpb: {baseline_bpb:.6f}")

    no_improve_count = 0
    agent = ClaudeCodeAgent(model=MODEL, timeout_sec=600)

    if topic is None:
        topic = (
            "你是 AI 研究员，正在对 GPT 模型训练代码 train.py 进行实验优化。"
            "目标：降低 val_bpb（validation bits per byte）。"
            "val_bpb 越低越好。每轮请提出一个具体的改进思路并直接修改 train.py。"
        )

    # 打印 header
    print()
    print("=== AutoResearch Loop Runner ===")
    print(
        f"baseline: {baseline_bpb:.6f} | max_iter: {max_iterations} | early_stop: {early_stop} | time_budget: {time_budget}s"
    )
    print("─" * 50)

    # 把 results.tsv 中的历史记录同步到 history.jsonl（去重）
    _sync_history_from_results_tsv()

    for iteration in range(1, max_iterations + 1):
        phase = "generate" if iteration == 1 else "refine"

        # === 阶段1: LLM 生成/ refine ===
        _print_progress(iteration, max_iterations, f"LLM ({phase})", "start")
        t0 = time.monotonic()

        train_py_content = (WORKDIR / "train.py").read_text()
        run_summaries = _build_run_summaries()
        result = agent.refine(
            current_files={"train.py": train_py_content},
            run_summaries=run_summaries,
            metric_key="val_bpb",
            metric_direction="minimize",
            topic=topic,
            extra_hints="基于历史结果和当前代码，选择一个最可能降低 val_bpb 的改进方向。",
            workdir=WORKDIR,
        )

        elapsed = time.monotonic() - t0

        # 写 LLM 回复日志
        _write_llm_log(iteration, phase, result.content)
        tsv_description = _extract_description_from_llm_reply(
            result.content,
            iteration=iteration,
        )

        if not result.success:
            _print_progress(
                iteration,
                max_iterations,
                f"LLM ({phase})",
                "fail",
                f"({elapsed:.1f}s) rc={result.rc}",
            )
            print(f"       stderr: {result.stderr[:200]}")
            continue

        _print_progress(
            iteration, max_iterations, f"LLM ({phase})", "done", f"({elapsed:.1f}s)"
        )

        # === 仅接受工作区文件结果，不再解析对话文本代码 ===
        new_code = result.files.get("train.py", "").strip()
        if not new_code:
            print(f"       WARNING: train.py was not updated by file tools")
            continue

        (WORKDIR / "train.py").write_text(new_code)

        # === Git commit ===
        desc = f"iteration {iteration}"
        git_hash = _git_commit("candidate", desc)

        # === 阶段2: 运行实验 ===
        _print_progress(iteration, max_iterations, "Run experiment", "start")

        exp_result = run_experiment(
            train_py_path=WORKDIR / "train.py",
            time_budget=time_budget,
        )

        runtime_min = exp_result.runtime_seconds / 60
        _print_progress(
            iteration, max_iterations, "Run experiment", "done", f"({runtime_min:.1f}m)"
        )

        # 失败后给一次“诊断+修复+重跑”的机会，避免直接丢弃可修复的实验
        if exp_result.status != "completed":
            repaired_result, repaired, repair_git_hash, repair_desc = _repair_once_on_failure(
                agent=agent,
                iteration=iteration,
                max_iterations=max_iterations,
                exp_result=exp_result,
                time_budget=time_budget,
            )
            if repaired:
                exp_result = repaired_result
                if repair_git_hash:
                    git_hash = repair_git_hash
                if repair_desc:
                    tsv_description = repair_desc

        # === 判定 ===
        if exp_result.status == "completed" and exp_result.val_bpb is not None:
            improved, decision = metric_judge(baseline_bpb, exp_result.val_bpb)
            if improved:
                baseline_bpb = exp_result.val_bpb
                no_improve_count = 0
        else:
            improved = False
            decision = "discard"

        # === 追加 results.tsv ===
        _append_results_tsv(
            git_commit_hash=git_hash,
            exp_result=exp_result,
            decision=decision,
            iteration=iteration,
            description=tsv_description,
        )

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        # === 落盘产物 ===
        _write_iter_artifacts(iteration, exp_result, improved, decision, git_hash)
        _append_history(iteration, exp_result, improved, decision, git_hash)
        _update_best_candidate(iteration, exp_result, git_hash)

        # === 打印结果行 ===
        if exp_result.val_bpb is not None:
            delta = baseline_bpb - exp_result.val_bpb
            delta_str = f"+{delta:.6f}" if improved else f"{delta:.6f}"
            print(f"       val_bpb: {exp_result.val_bpb:.6f}  improved: {delta_str}")
        else:
            print(f"       val_bpb: None  status: {exp_result.status}")
        print(f"       status: {exp_result.status} | decision: {decision}")
        print(f"       artifacts/iter-{iteration:03d}.json written")
        print(f"       llm_iter-{iteration:03d}_{phase}.log written")
        print(f"       run.log written")
        print()

        # === 早停检查 ===
        if no_improve_count >= early_stop:
            print(f"[loop_runner] Early stop: no improvement for {early_stop} rounds")
            _write_run_summary("early_stop", baseline_bpb)
            break
    else:
        _write_run_summary("max_iterations_reached", baseline_bpb)

    # === 最终汇总 ===
    print()
    print("=== Loop finished ===")
    reason = (
        "early_stop" if no_improve_count >= early_stop else "max_iterations_reached"
    )
    print(f"stop_reason: {reason}")
    print(f"best_bpb: {baseline_bpb:.6f}")
    print(f"artifacts: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoResearch Loop Runner")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--topic", type=str, default=None)
    args = parser.parse_args()

    run_loop(
        max_iterations=args.max_iterations,
        early_stop=args.early_stop,
        time_budget=args.time_budget,
        topic=args.topic,
    )
