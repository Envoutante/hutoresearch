"""loop_runner: 自动循环实验主循环"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# 支持直接运行: python autorunner/loop_runner.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.claude_code_agent import ClaudeCodeAgent
from autorunner.experiment_executor import (
    ExperimentResult,
    analyze_failure,
    judge as metric_judge,
    run as run_experiment,
)


WORKDIR = Path("/mount/disk1/rl-hyr/autoresearch")
ARTIFACTS_DIR = WORKDIR / "autorunner" / "artifacts"
RUN_LOG_FILE = ARTIFACTS_DIR / "run.log"
CURRENT_STATE_FILE = ARTIFACTS_DIR / "current_state.md"
RESULTS_TSV_FILE = WORKDIR / "results.tsv"
BEST_CANDIDATE_FILE = ARTIFACTS_DIR / "best_candidate.json"
MODEL = "MiniMax-M2.7"
console = Console()
_nvidia_live: Live | None = None


def _clear_nvidia_smi_panel():
    global _nvidia_live
    if _nvidia_live is None:
        return
    try:
        _nvidia_live.stop()
    except Exception:
        pass
    _nvidia_live = None


def print_auto_research_banner(
    model: str | None = None,
    provider: str | None = None,
    mode: str | None = None,
):
    logo_lines = [
        " █████╗ ██╗   ██╗████████╗ ██████╗     ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗",
        "██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║",
        "███████║██║   ██║   ██║   ██║   ██║    ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║",
        "██╔══██║██║   ██║   ██║   ██║   ██║    ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║",
        "██║  ██║╚██████╔╝   ██║   ╚██████╔╝    ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║",
        "╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝",
    ]
    gradient = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]

    for line, color in zip(logo_lines, gradient, strict=False):
        console.print(Text(line, style=f"bold {color}"))

    console.print(
        Text("  Auto Research · Autonomous Discovery Engine", style="dim italic")
    )

    info_parts: list[tuple[str, str]] = []
    if model:
        info_parts.append(("Model", model))
    if provider:
        info_parts.append(("Provider", provider))
    if mode:
        info_parts.append(("Mode", mode))

    if info_parts:
        info = Text("  ", style="dim")
        for i, (k, v) in enumerate(info_parts):
            if i > 0:
                info.append("  ", style="dim")
            info.append(f"{k}: ", style="dim")
            info.append(v, style="magenta")
        console.print(info)


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


def _load_best_keep_record() -> tuple[str | None, float | None]:
    """从 results.tsv 读取 status=keep 的最佳记录（commit, val_bpb）。"""
    results_tsv = WORKDIR / "results.tsv"
    if not results_tsv.exists():
        return None, None

    best_commit = None
    best_bpb = None
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
        except ValueError:
            continue
        if best_bpb is None or bpb < best_bpb:
            best_bpb = bpb
            best_commit = parts[0].strip() or None

    return best_commit, best_bpb


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


def _git_head_short() -> str:
    """读取当前 HEAD 短哈希。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _restore_train_py_from_commit(commit_hash: str) -> bool:
    """仅将 train.py 恢复到指定 commit，不影响其他文件。"""
    if not commit_hash:
        return False
    try:
        subprocess.run(
            ["git", "restore", "--source", commit_hash, "--worktree", "train.py"],
            cwd=WORKDIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _write_llm_log(iteration: int, phase: str, content: str):
    """LLM 回复写入日志文件"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    llm_log = ARTIFACTS_DIR / f"llm_iter-{iteration:03d}_{phase}.log"
    llm_log.write_text(content)


def _print_progress(
    iteration: int, max_iter: int, label: str, status: str, extra: str = ""
):
    """终端打印一行进度"""
    if status == "start" and not label.startswith("Run experiment"):
        _clear_nvidia_smi_panel()

    symbols = {"start": "⏳", "done": "🗸", "fail": "🗴", "skip": "»"}
    sym = symbols.get(status, "  ")
    line = Text()
    line.append(f"[{iteration}/{max_iter}] ", style="bold cyan")
    line.append(f"{label} ", style="bold")
    line.append(f"{sym}", style="yellow")
    if extra:
        line.append(f" {extra}", style="dim")
    console.print(line)


def _print_nvidia_smi_snapshot(pid: int):
    """训练进程启动后打印一次 nvidia-smi 快照，辅助确认实验是否正常运行。"""
    global _nvidia_live
    title = Text("nvidia-smi", style="bold cyan")
    title.append(f" (after start, pid={pid})", style="dim")

    max_attempts = 6
    retry_interval_sec = 1.0

    def _extract_processes_table(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        proc_idx = -1
        for i, line in enumerate(lines):
            if "Processes:" in line:
                proc_idx = i
                break
        if proc_idx < 0:
            return ""

        start = proc_idx
        for i in range(proc_idx, -1, -1):
            if lines[i].lstrip().startswith("+"):
                start = i
                break

        end = len(lines) - 1
        for i in range(proc_idx + 1, len(lines)):
            if lines[i].lstrip().startswith("+"):
                end = i
                break

        return "\n".join(lines[start : end + 1]).strip()

    def _extract_numeric_tokens(text: str) -> set[int]:
        nums: set[int] = set()
        for token in text.replace("|", " ").split():
            if token.isdigit():
                nums.add(int(token))
        return nums

    def _get_child_pids(parent_pid: int) -> set[int]:
        try:
            result = subprocess.run(
                ["ps", "-o", "pid=", "--ppid", str(parent_pid)],
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return set()

        children: set[int] = set()
        for raw in (result.stdout or "").splitlines():
            s = raw.strip()
            if s.isdigit():
                children.add(int(s))
        return children

    output = ""
    err = ""
    result_code = 0
    timeout_happened = False
    target_visible = False
    processes_table = ""

    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=12,
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    Text("nvidia-smi command not found", style="yellow"),
                    title=title,
                    border_style="yellow",
                )
            )
            return
        except subprocess.TimeoutExpired:
            timeout_happened = True
            break

        output = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        result_code = result.returncode

        if result_code != 0 and err:
            break

        processes_table = _extract_processes_table(output)
        if processes_table:
            running_ids = _extract_numeric_tokens(processes_table)
            target_ids = {pid} | _get_child_pids(pid)
            if target_ids & running_ids:
                target_visible = True
                break

        if attempt < max_attempts:
            time.sleep(retry_interval_sec)

    if timeout_happened:
        body = Text("nvidia-smi timed out", style="yellow")
        border = "yellow"
    elif result_code != 0 and err:
        body = Text(err[:2000], style="red")
        border = "red"
    elif not output:
        body = Text("(no output)", style="dim")
        border = "yellow"
    elif processes_table:
        if target_visible:
            body = Text(processes_table[:5000])
            border = "blue"
        else:
            note = f"target pid {pid} not visible after {max_attempts} checks\n\n"
            body = Text((note + processes_table)[:5000], style="yellow")
            border = "yellow"
    else:
        body = Text(
            f"(Processes table not found after {max_attempts} checks)",
            style="yellow",
        )
        border = "yellow"

    _clear_nvidia_smi_panel()
    panel = Panel(body, title=title, border_style=border)
    try:
        live = Live(panel, console=console, refresh_per_second=4, transient=True)
        live.start()
        _nvidia_live = live
    except Exception:
        console.print(panel)


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


def _latest_eval_file() -> Path | None:
    """返回 artifacts 下最新的 eval-*.json。"""
    if not ARTIFACTS_DIR.exists():
        return None

    files = sorted(ARTIFACTS_DIR.glob("eval-*.json"))
    return files[-1] if files else None


def _build_eval_guidance() -> str:
    """构造给 Generator 的评估上下文读取提示。"""
    latest_eval = _latest_eval_file()
    if latest_eval is None:
        return (
            "请先读取 autorunner/artifacts/current_state.md（若存在）恢复上下文。"
            "当前没有 eval-*.json，请基于 results.tsv 与 current_state 制定方向。"
        )

    return (
        "请先读取 autorunner/artifacts/current_state.md（若存在）恢复上下文。"
        f"最新评估文件：autorunner/artifacts/{latest_eval.name}。"
        "必须重点参考 recommendation.next_action 与 diagnosis，再决定本轮方案。"
    )


def _write_iter_artifacts(
    iteration: int,
    exp_result: ExperimentResult,
    improved: bool,
    decision: str,
    git_commit_hash: str,
    description: str,
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
        "notes": description,
    }
    iter_file = ARTIFACTS_DIR / f"iter-{iteration:03d}.json"
    iter_file.write_text(json.dumps(entry, indent=2))


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
    peak_vram_mb = (
        exp_result.peak_vram_mb if exp_result.peak_vram_mb is not None else 0.0
    )
    memory_gb = peak_vram_mb / 1024.0
    status = "keep" if decision == "keep" else "discard"
    line = (
        f"{git_commit_hash}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    )
    with results_tsv.open("a") as f:
        f.write(line)


def _extract_description_from_llm_reply(
    reply: str, iteration: int, prefix: str = ""
) -> str:
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


def _extract_global_upper_assignments(code: str) -> dict[str, str]:
    """提取 train.py 中全大写常量赋值（用于 repair 约束校验）。"""
    out: dict[str, str] = {}
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        name = left.strip()
        if not name or not name.isidentifier() or not name.isupper():
            continue
        out[name] = right.strip()
    return out


def _detect_forbidden_repair_changes(before_code: str, after_code: str) -> list[str]:
    """检测 repair 是否改动了优化相关超参/结构参数（全大写常量）。"""
    before = _extract_global_upper_assignments(before_code)
    after = _extract_global_upper_assignments(after_code)

    changed: list[str] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changed.append(key)
    return changed


def _repair_once_on_failure(
    *,
    agent: ClaudeCodeAgent,
    iteration: int,
    max_iterations: int,
    exp_result: ExperimentResult,
    time_budget: int,
    run_log_path: Path,
    pre_iteration_code: str,
    refine_description: str,
) -> tuple[ExperimentResult, bool, str | None, str | None]:
    """失败时最多修复 3 次并重跑；均失败则回退到本轮开始前代码。"""
    max_repair_attempts = 3
    current_result = exp_result
    last_repair_git_hash: str | None = None
    last_repair_desc: str | None = None
    attempted_repairs = False

    for attempt in range(1, max_repair_attempts + 1):
        # 仅对可修复失败（依赖/运行时）触发重试，避免 timeout/数值异常死循环
        analysis = analyze_failure(
            current_result.failure_type or "runtime", current_result.log_tail
        )
        if not analysis.should_retry:
            break

        attempted_repairs = True
        stage = f"LLM (repair {attempt}/{max_repair_attempts})"
        _print_progress(iteration, max_iterations, stage, "start")
        t0 = time.monotonic()

        current_code = (WORKDIR / "train.py").read_text()
        issues = (
            f"failure_type: {analysis.failure_type}\n"
            f"repair_strategy: {analysis.repair_strategy}\n\n"
            "错误日志尾部（最多 50 行）：\n"
            f"{current_result.log_tail}"
        )
        repair_result = agent.repair(
            files={"train.py": current_code},
            issues=issues,
            refine_description=refine_description,
            workdir=WORKDIR,
        )

        elapsed = time.monotonic() - t0
        _write_llm_log(iteration, "repair", repair_result.content)

        if not repair_result.success:
            _print_progress(
                iteration,
                max_iterations,
                stage,
                "fail",
                f"({elapsed:.1f}s) rc={repair_result.rc}",
            )
            err = (repair_result.stderr or repair_result.content or "").strip()
            if err:
                console.print(Text(f"       error: {err[:300]}", style="red"))
            continue

        # 仅接受 Claude 在工作区原地修改后的 train.py
        repaired_code = repair_result.files.get("train.py", "").strip()
        if not repaired_code:
            _print_progress(
                iteration,
                max_iterations,
                stage,
                "fail",
                "(train.py not updated)",
            )
            continue

        if repaired_code == current_code:
            _print_progress(
                iteration,
                max_iterations,
                stage,
                "skip",
                "(no code change)",
            )
            continue

        forbidden_changes = _detect_forbidden_repair_changes(
            current_code, repaired_code
        )
        if forbidden_changes:
            changed_keys = ", ".join(forbidden_changes[:8])
            if len(forbidden_changes) > 8:
                changed_keys += ", ..."
            _print_progress(
                iteration,
                max_iterations,
                stage,
                "fail",
                "(forbidden optimization/architecture changes)",
            )
            console.print(Text(f"       blocked keys: {changed_keys}", style="red"))
            continue

        (WORKDIR / "train.py").write_text(repaired_code)
        last_repair_git_hash = _git_commit("repair", f"iteration {iteration}")
        _print_progress(iteration, max_iterations, stage, "done", f"({elapsed:.1f}s)")

        _print_progress(
            iteration,
            max_iterations,
            f"Run experiment (after repair {attempt}/{max_repair_attempts})",
            "start",
        )
        rerun_result = run_experiment(
            train_py_path=WORKDIR / "train.py",
            time_budget=time_budget,
            run_log_path=run_log_path,
            on_process_started=_print_nvidia_smi_snapshot,
        )
        runtime_min = rerun_result.runtime_seconds / 60
        _print_progress(
            iteration,
            max_iterations,
            f"Run experiment (after repair {attempt}/{max_repair_attempts})",
            "done",
            f"({runtime_min:.1f}m)",
        )

        # repair 成功后，允许其基于 refine 描述重写“最终实验方向描述”，
        # 但若输出明显偏向报错排障语义，则回退到 refine 描述。
        candidate_desc = _extract_description_from_llm_reply(
            repair_result.content,
            iteration=iteration,
        )
        low = candidate_desc.lower()
        if (
            candidate_desc == f"iter {iteration}"
            or "repair" in low
            or "bug" in low
            or "报错" in candidate_desc
            or "错误" in candidate_desc
            or "修复" in candidate_desc
            or "异常" in candidate_desc
        ):
            last_repair_desc = refine_description
        else:
            last_repair_desc = candidate_desc

        if rerun_result.status == "completed":
            return rerun_result, True, last_repair_git_hash, last_repair_desc

        current_result = rerun_result

    # 3 次修复均失败：回退到本轮迭代开始前代码，避免带着坏代码进入下一轮。
    if attempted_repairs:
        current_code = (WORKDIR / "train.py").read_text()
        if current_code != pre_iteration_code:
            (WORKDIR / "train.py").write_text(pre_iteration_code)
            print("       rollback: train.py restored to pre-iteration version")

    return current_result, False, None, None


def _run_evaluator(
    *,
    agent: ClaudeCodeAgent,
    iteration: int,
    max_iterations: int,
) -> Path | None:
    """单轮实验结束后调用独立 Evaluator，产出 eval-<n>.json 与 current_state.md。"""
    iter_artifact = ARTIFACTS_DIR / f"iter-{iteration:03d}.json"
    eval_artifact = ARTIFACTS_DIR / f"eval-{iteration:03d}.json"

    _print_progress(iteration, max_iterations, "LLM (evaluate)", "start")
    t0 = time.monotonic()

    train_before = (WORKDIR / "train.py").read_text()
    eval_result = agent.evaluate(
        iteration=iteration,
        iter_artifact_path=iter_artifact,
        run_log_path=RUN_LOG_FILE,
        current_state_path=CURRENT_STATE_FILE,
        results_tsv_path=RESULTS_TSV_FILE,
        best_candidate_path=BEST_CANDIDATE_FILE,
        eval_output_path=eval_artifact,
        workdir=WORKDIR,
    )
    elapsed = time.monotonic() - t0
    _write_llm_log(iteration, "evaluate", eval_result.content)

    # Evaluator 只负责分析，不应改写 train.py；若误改则回滚。
    train_after = (WORKDIR / "train.py").read_text()
    if train_after != train_before:
        (WORKDIR / "train.py").write_text(train_before)
        print("       WARNING: evaluator modified train.py, reverted.")

    if not eval_result.success and not eval_artifact.exists():
        _print_progress(
            iteration,
            max_iterations,
            "LLM (evaluate)",
            "fail",
            f"({elapsed:.1f}s) rc={eval_result.rc}",
        )
        err = (eval_result.stderr or eval_result.content or "").strip()
        if err:
            console.print(Text(f"       error: {err[:300]}", style="red"))
        return None

    if not eval_artifact.exists():
        _print_progress(
            iteration,
            max_iterations,
            "LLM (evaluate)",
            "fail",
            "(eval json missing)",
        )
        return None

    try:
        json.loads(eval_artifact.read_text())
    except json.JSONDecodeError as exc:
        _print_progress(
            iteration,
            max_iterations,
            "LLM (evaluate)",
            "fail",
            f"(invalid eval json: {exc})",
        )
        return None

    _print_progress(
        iteration,
        max_iterations,
        "LLM (evaluate)",
        "done",
        f"({elapsed:.1f}s)",
    )

    if not CURRENT_STATE_FILE.exists():
        print("       WARNING: current_state.md was not updated by evaluator")

    return eval_artifact


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
        console.print(
            Text(
                "[loop_runner] No baseline found, starting from scratch", style="yellow"
            )
        )
    else:
        line = Text("[loop_runner] ", style="dim")
        line.append("Baseline val_bpb: ", style="bold")
        line.append(f"{baseline_bpb:.6f}", style="bold green")
        console.print(line)

    no_improve_count = 0
    agent = ClaudeCodeAgent(model=MODEL, timeout_sec=600)
    best_commit, best_bpb_record = _load_best_keep_record()

    if topic is None:
        topic = (
            "你是 AI 研究员，正在对 GPT 模型训练代码 train.py 进行实验优化。"
            "目标：降低 val_bpb（validation bits per byte）。"
            "val_bpb 越低越好。每轮请提出一个具体的改进思路并直接修改 train.py。"
        )

    # 打印 header
    console.print()
    print_auto_research_banner(model=MODEL, provider="Claude Code CLI", mode="loop")
    console.print(Text("=== AutoResearch Loop Runner ===", style="bold white"))
    summary = Text("baseline: ", style="dim")
    summary.append(f"{baseline_bpb:.6f}", style="bold green")
    summary.append(" | ", style="dim")
    summary.append("max_iter: ", style="dim")
    summary.append(str(max_iterations), style="bold")
    summary.append(" | ", style="dim")
    summary.append("early_stop: ", style="dim")
    summary.append(str(early_stop), style="bold")
    summary.append(" | ", style="dim")
    summary.append("time_budget: ", style="dim")
    summary.append(f"{time_budget}s", style="bold")
    console.print(summary)
    console.print(Text("─" * 50, style="dim"))

    for iteration in range(1, max_iterations + 1):
        phase = "bootstrap" if iteration == 1 else "refine"
        train_py_content = (WORKDIR / "train.py").read_text()
        git_hash = ""
        tsv_description = f"iter {iteration}"
        llm_log_written = False

        if iteration == 1:
            # === bootstrap: iter=1 ===
            if best_bpb_record is None or not best_commit:
                _print_progress(
                    iteration,
                    max_iterations,
                    "LLM (bootstrap)",
                    "skip",
                    "(no history; baseline directly)",
                )
                git_hash = _git_head_short()
                tsv_description = "baseline(no_history)"
            else:
                restored = _restore_train_py_from_commit(best_commit)
                if restored:
                    _print_progress(
                        iteration,
                        max_iterations,
                        "Bootstrap restore",
                        "done",
                        f"(train.py <- {best_commit})",
                    )
                else:
                    _print_progress(
                        iteration,
                        max_iterations,
                        "Bootstrap restore",
                        "fail",
                        f"(restore {best_commit} failed)",
                    )

                # 可选分析：仅总结历史，不改代码
                _print_progress(iteration, max_iterations, "LLM (bootstrap)", "start")
                t0 = time.monotonic()
                eval_guidance = _build_eval_guidance()
                exp_plan = (
                    "分析并总结历史实验结果，为后续 refine 提供可执行方向；"
                    "本轮不产出代码改动。"
                )
                extra_guidance = (
                    "请仅做只读分析：总结历史有效方向、失败模式与下一轮优先级。"
                    "禁止修改 train.py。"
                    f"历史最佳参考：commit={best_commit}, val_bpb={best_bpb_record:.6f}。"
                    f"{eval_guidance}"
                )
                result = agent.generate(
                    exp_plan=exp_plan,
                    topic=topic,
                    metric_key="val_bpb",
                    extra_guidance=extra_guidance,
                    workdir=WORKDIR,
                )
                elapsed = time.monotonic() - t0
                _write_llm_log(iteration, "generate", result.content)
                llm_log_written = True

                if result.success:
                    _print_progress(
                        iteration,
                        max_iterations,
                        "LLM (bootstrap)",
                        "done",
                        f"({elapsed:.1f}s)",
                    )
                    tsv_description = _extract_description_from_llm_reply(
                        result.content,
                        iteration=iteration,
                        prefix="bootstrap_analysis: ",
                    )
                else:
                    _print_progress(
                        iteration,
                        max_iterations,
                        "LLM (bootstrap)",
                        "fail",
                        f"({elapsed:.1f}s) rc={result.rc}",
                    )
                    tsv_description = f"baseline(from_best:{best_commit})"

                # 无论分析是否成功，都使用 best commit 作为本轮基线来源标识
                git_hash = best_commit
        else:
            # === 阶段1: LLM refine ===
            _print_progress(iteration, max_iterations, f"LLM ({phase})", "start")
            t0 = time.monotonic()

            run_summaries = _build_run_summaries()
            eval_guidance = _build_eval_guidance()
            result = agent.refine(
                current_files={"train.py": train_py_content},
                run_summaries=run_summaries,
                metric_key="val_bpb",
                metric_direction="minimize",
                topic=topic,
                extra_hints=(
                    "基于历史结果和当前代码，选择一个最可能降低 val_bpb 的改进方向。"
                    f"{eval_guidance}"
                ),
                workdir=WORKDIR,
            )

            elapsed = time.monotonic() - t0

            # 写 LLM 回复日志
            _write_llm_log(iteration, phase, result.content)
            llm_log_written = True
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
                err = (result.stderr or result.content or "").strip()
                if err:
                    console.print(Text(f"       error: {err[:300]}", style="red"))
                else:
                    console.print(
                        Text(
                            "       error: (empty) see autorunner/artifacts/live_stream_*.jsonl",
                            style="red",
                        )
                    )
                continue

            _print_progress(
                iteration,
                max_iterations,
                f"LLM ({phase})",
                "done",
                f"({elapsed:.1f}s)",
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
            run_log_path=RUN_LOG_FILE,
            on_process_started=_print_nvidia_smi_snapshot,
        )

        runtime_min = exp_result.runtime_seconds / 60
        _print_progress(
            iteration, max_iterations, "Run experiment", "done", f"({runtime_min:.1f}m)"
        )

        # 失败后给一次“诊断+修复+重跑”的机会，避免直接丢弃可修复的实验
        if exp_result.status != "completed":
            repaired_result, repaired, repair_git_hash, repair_desc = (
                _repair_once_on_failure(
                    agent=agent,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    exp_result=exp_result,
                    time_budget=time_budget,
                    run_log_path=RUN_LOG_FILE,
                    pre_iteration_code=train_py_content,
                    refine_description=tsv_description,
                )
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
        _write_iter_artifacts(
            iteration,
            exp_result,
            improved,
            decision,
            git_hash,
            tsv_description,
        )
        _update_best_candidate(iteration, exp_result, git_hash)

        eval_artifact = _run_evaluator(
            agent=agent,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        # === 打印结果行 ===
        if exp_result.val_bpb is not None:
            delta = baseline_bpb - exp_result.val_bpb
            delta_str = f"+{delta:.6f}" if improved else f"{delta:.6f}"
            print(f"       val_bpb: {exp_result.val_bpb:.6f}  improved: {delta_str}")
        else:
            print(f"       val_bpb: None  status: {exp_result.status}")
        print(f"       status: {exp_result.status} | decision: {decision}")
        print(f"       artifacts/iter-{iteration:03d}.json written")
        if eval_artifact is not None:
            print(f"       {eval_artifact.relative_to(WORKDIR)} written")
        if llm_log_written:
            print(f"       llm_iter-{iteration:03d}_{phase}.log written")
        print(f"       {RUN_LOG_FILE.relative_to(WORKDIR)} written")
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
