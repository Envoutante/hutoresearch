"""parallel_runner: 并行候选生成与实验调度（第一版，单文件实现）"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# 支持直接运行: python autorunner/parallel_runner.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.claude_code_agent import ClaudeCodeAgent
from autorunner.env_config import load_dotenv, project_root
from autorunner.embedding import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DescriptionEmbeddingIndex,
)
from autorunner.experiment_executor import (
    ExperimentResult,
    analyze_failure,
    judge as metric_judge,
    run,
)

PROJECT_ROOT = project_root()
load_dotenv()

WORKDIR = Path(os.getenv("AR_WORKDIR", str(PROJECT_ROOT))).expanduser()
ARTIFACTS_DIR = WORKDIR / "autorunner" / "artifacts"
CANDIDATE_ROOT = WORKDIR / "autorunner" / "candidates"
RESULTS_TSV_FILE = WORKDIR / "results.tsv"
FAILURE_DIRECTIONS_FILE = ARTIFACTS_DIR / "failure_directions.json"
QUEUE_EVENTS_FILE = ARTIFACTS_DIR / "parallel_queue.jsonl"
BEST_TRAIN_FILE = ARTIFACTS_DIR / "best_train.py"
CURRENT_STATE_FILE = ARTIFACTS_DIR / "current_state.md"
MODEL = os.getenv("AR_MODEL", "deepseek-v4-pro[1m]")
console = Console()


@dataclass
class CandidateTask:
    candidate_id: str
    workdir: Path
    train_py_path: Path
    refine_description: str
    parent_ref: str
    created_at: float
    queued_at: float
    status: str = "queued"
    gpu_id: int | None = None
    launch_ts: float | None = None
    final_description: str = ""


@dataclass
class RunningTask:
    task: CandidateTask
    future: Future[ExperimentResult]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _append_queue_event(event_type: str, payload: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now_iso(),
        "event": event_type,
        "payload": payload,
    }
    with QUEUE_EVENTS_FILE.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _short_text(text: str, max_len: int = 42) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _load_recent_queue_events(limit: int = 8) -> list[str]:
    if not QUEUE_EVENTS_FILE.exists():
        return []

    rows: deque[str] = deque(maxlen=limit)
    try:
        with QUEUE_EVENTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                ts = str(obj.get("ts") or "")
                ev = str(obj.get("event") or "unknown")
                payload = (
                    obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
                )
                cid = str(payload.get("candidate_id") or "")
                gpu = payload.get("gpu_id")
                attempt = payload.get("attempt")
                extra = []
                if cid:
                    extra.append(cid)
                if gpu is not None:
                    extra.append(f"gpu={gpu}")
                if attempt is not None:
                    extra.append(f"try={attempt}")
                tail = f" ({', '.join(extra)})" if extra else ""
                rows.append(f"{ts} | {ev}{tail}")
    except OSError:
        return []
    return list(rows)


def _render_dashboard(
    *,
    start_ts: float,
    completed_runs: int,
    max_total_runs: int,
    baseline_bpb: float,
    queued: list[CandidateTask],
    running: dict[str, RunningTask],
    finished_recent: list[dict[str, str]],
    repair_stats: dict[str, int],
    failure_reason_counts: dict[str, int],
    dashboard_status_filter: str,
    mem_idle_threshold_mb: int,
    util_idle_threshold_pct: int,
) -> Panel:
    elapsed = int(time.time() - start_ts)
    gpu_snapshots = _gpu_snapshots()

    summary = Table.grid(expand=True)
    summary.add_column(ratio=1)
    summary.add_column(ratio=1)
    baseline_text = "inf" if baseline_bpb == float("inf") else f"{baseline_bpb:.6f}"
    summary.add_row(
        f"elapsed: {elapsed}s | progress: {completed_runs}/{max_total_runs}",
        f"queued: {len(queued)} | running: {len(running)} | baseline: {baseline_text}",
    )
    summary.add_row(
        (
            "repair: "
            f"start={repair_stats.get('start', 0)} "
            f"success={repair_stats.get('success', 0)} "
            f"failed={repair_stats.get('failed', 0)} "
            f"out_of_scope={repair_stats.get('out_of_scope', 0)}"
        ),
        f"filter: {dashboard_status_filter}",
    )

    gpu_table = Table(title="GPU", expand=True)
    gpu_table.add_column("gpu", style="cyan", justify="right")
    gpu_table.add_column("mem(MiB)", justify="right")
    gpu_table.add_column("util(%)", justify="right")
    gpu_table.add_column("state")
    gpu_table.add_column("candidate")

    running_by_gpu: dict[int, str] = {}
    for rt in running.values():
        if rt.task.gpu_id is not None:
            running_by_gpu[rt.task.gpu_id] = rt.task.candidate_id

    for gpu_id, mem_used, util in gpu_snapshots:
        cid = running_by_gpu.get(gpu_id, "")
        busy = bool(cid)
        state = "running" if busy else "idle"
        state_style = "green" if busy else "dim"
        mem_style = "green" if mem_used <= mem_idle_threshold_mb else "red"
        util_style = "green" if util <= util_idle_threshold_pct else "red"
        gpu_table.add_row(
            str(gpu_id),
            f"[{mem_style}]{mem_used}[/{mem_style}]/{mem_idle_threshold_mb}",
            f"[{util_style}]{util}[/{util_style}]/{util_idle_threshold_pct}",
            f"[{state_style}]{state}[/{state_style}]",
            cid,
        )

    cand_table = Table(title="Candidates", expand=True)
    cand_table.add_column("id", style="cyan")
    cand_table.add_column("status")
    cand_table.add_column("gpu", justify="right")
    cand_table.add_column("wait(s)", justify="right")
    cand_table.add_column("runtime(s)", justify="right")
    cand_table.add_column("desc")

    def _allow_status(status: str) -> bool:
        if dashboard_status_filter == "all":
            return True
        if dashboard_status_filter == "active":
            return status in {"queued", "running", "repairing"}
        if dashboard_status_filter == "failed":
            return status in {"discard", "failed"}
        if dashboard_status_filter == "finished":
            return status in {"keep", "discard"}
        return True

    now = time.time()
    for task in queued[-20:]:
        if not _allow_status("queued"):
            continue
        wait_s = int(max(0, now - task.queued_at))
        cand_table.add_row(
            task.candidate_id,
            "[yellow]queued[/yellow]",
            "-",
            str(wait_s),
            "-",
            _short_text(task.refine_description),
        )

    for rt in running.values():
        t = rt.task
        current_status = t.status if t.status else "running"
        if not _allow_status(current_status):
            continue
        run_s = int(max(0, now - (t.launch_ts or now)))
        status_text = (
            "[magenta]repairing[/magenta]"
            if current_status == "repairing"
            else "[blue]running[/blue]"
        )
        cand_table.add_row(
            t.candidate_id,
            status_text,
            str(t.gpu_id) if t.gpu_id is not None else "-",
            str(int(max(0, now - t.queued_at))),
            str(run_s),
            _short_text(t.refine_description),
        )

    for row in finished_recent[-20:]:
        status = row.get("status", "")
        if not _allow_status(status):
            continue
        status_style = "green" if status == "keep" else "red"
        cand_table.add_row(
            row.get("candidate_id", ""),
            f"[{status_style}]{status}[/{status_style}]",
            row.get("gpu", "-"),
            row.get("wait", "-"),
            row.get("runtime", "-"),
            _short_text(row.get("desc", "")),
        )

    reason_table = Table(title="Failure Reasons", expand=True)
    reason_table.add_column("reason")
    reason_table.add_column("count", justify="right")
    for reason, count in sorted(
        failure_reason_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:8]:
        reason_table.add_row(reason, str(count))

    event_table = Table(title="Recent Events", expand=True)
    event_table.add_column("event", style="dim")
    for line in _load_recent_queue_events(limit=8):
        event_table.add_row(line)

    group = Group(summary, gpu_table, cand_table, reason_table, event_table)
    return Panel(group, title="AutoResearch Parallel Dashboard", border_style="cyan")


def _load_baseline() -> float:
    if not RESULTS_TSV_FILE.exists():
        return float("inf")

    best = None
    for line in RESULTS_TSV_FILE.read_text().splitlines():
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
        if best is None or bpb < best:
            best = bpb

    return best if best is not None else float("inf")


def _build_run_summaries() -> list[str]:
    if not RESULTS_TSV_FILE.exists():
        return []

    summaries: list[str] = []
    for line in RESULTS_TSV_FILE.read_text().splitlines():
        if line.startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        summaries.append(
            f"commit={parts[0].strip()} val_bpb={parts[1].strip()} status={parts[3].strip()} desc={parts[4].strip()}"
        )
    return summaries


def _latest_eval_file() -> Path | None:
    if not ARTIFACTS_DIR.exists():
        return None
    files = sorted(ARTIFACTS_DIR.glob("eval-*.json"))
    return files[-1] if files else None


def _build_eval_guidance() -> str:
    latest_eval = _latest_eval_file()
    if latest_eval is None:
        return (
            "请先读取 autorunner/artifacts/current_state.md（若存在）恢复上下文。"
            "当前没有 eval-*.json，请基于 results.tsv、current_state 和 candidates 历史避免重复方向。"
        )

    return (
        "请先读取 autorunner/artifacts/current_state.md（若存在）恢复上下文。"
        f"最新评估文件：autorunner/artifacts/{latest_eval.name}。"
        "必须重点参考 recommendation.next_action 与 diagnosis，再决定本轮方案；"
        "若与近几轮方向相似，必须说明新增差异点。"
    )


def _to_outcome(exp_result: ExperimentResult) -> str:
    if exp_result.status == "timeout":
        return "timeout"
    if exp_result.status == "crashed":
        return "crashed"
    if exp_result.status != "completed":
        return "completed_anomaly"
    if exp_result.val_bpb is None:
        return "completed_anomaly"
    return "completed_success"


def _safe_relative_change_percent(
    baseline_before: float,
    val_bpb: float | None,
) -> float | None:
    if val_bpb is None:
        return None
    if not math.isfinite(baseline_before) or baseline_before == 0:
        return None
    return (val_bpb - baseline_before) / baseline_before * 100.0


def _write_parallel_eval_artifact(
    *,
    task: CandidateTask,
    exp_result: ExperimentResult,
    baseline_before: float,
    baseline_after: float,
    improved: bool,
    decision: str,
    repaired: bool,
    discard_reason: str | None,
) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = ARTIFACTS_DIR / f"eval-{task.candidate_id}.json"

    next_action = "continue" if improved else "discard_and_pivot"
    if exp_result.status in {"timeout", "crashed"}:
        next_action = "investigate"

    issues: list[str] = []
    if exp_result.status != "completed":
        issues.append(f"run_status={exp_result.status}")
    if discard_reason:
        issues.append(f"discard_reason={discard_reason}")
    if repaired:
        issues.append("candidate_required_repair")

    payload = {
        "candidate_id": task.candidate_id,
        "evaluated_at": _now_iso(),
        "outcome": _to_outcome(exp_result),
        "metrics": {
            "val_bpb": exp_result.val_bpb,
            "best_val_bpb_before": (
                baseline_before if math.isfinite(baseline_before) else None
            ),
            "best_val_bpb_after": (
                baseline_after if math.isfinite(baseline_after) else None
            ),
            "relative_change_percent": _safe_relative_change_percent(
                baseline_before,
                exp_result.val_bpb,
            ),
            "peak_vram_mb": exp_result.peak_vram_mb,
            "mfu_percent": exp_result.mfu_percent,
            "training_seconds": exp_result.training_seconds,
        },
        "metrics_healthy": bool(
            exp_result.status == "completed"
            and exp_result.val_bpb is not None
            and math.isfinite(exp_result.val_bpb)
        ),
        "diagnosis": {
            "summary": (
                "improved" if improved else "not_improved"
            ),
            "issues": issues,
            "root_cause": discard_reason or "n/a",
        },
        "code_review": {
            "diff_summary": task.final_description or task.refine_description,
            "risks": ["direction_similarity_risk"],
            "consistency": True,
        },
        "recommendation": {
            "next_action": next_action,
            "reasoning": (
                "Metric improved; keep exploring nearby variants."
                if improved
                else "Metric did not improve; pivot with explicit novelty."
            ),
            "suggested_directions": [
                "Propose a clearly differentiated direction from recent candidates.",
                "Reference current_state.md and eval artifacts before editing train.py.",
            ],
        },
        "decision": decision,
        "repaired": repaired,
    }
    eval_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return eval_path


def _append_parallel_current_state(
    *,
    task: CandidateTask,
    exp_result: ExperimentResult,
    decision: str,
    improved: bool,
    baseline_before: float,
    baseline_after: float,
    eval_path: Path,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## {task.candidate_id} @ {_now_iso()}",
        f"description: {task.final_description or task.refine_description}",
        f"status: {exp_result.status}",
        f"decision: {decision}",
        f"improved: {improved}",
        f"val_bpb: {exp_result.val_bpb}",
        (
            f"best_val_bpb_before: {baseline_before}"
            if math.isfinite(baseline_before)
            else "best_val_bpb_before: None"
        ),
        (
            f"best_val_bpb_after: {baseline_after}"
            if math.isfinite(baseline_after)
            else "best_val_bpb_after: None"
        ),
        f"eval_file: autorunner/artifacts/{eval_path.name}",
        "",
    ]

    mode = "a" if CURRENT_STATE_FILE.exists() else "w"
    with CURRENT_STATE_FILE.open(mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write("\n".join(lines))


def _extract_description_from_llm_reply(reply: str, fallback: str) -> str:
    if not reply:
        return fallback

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

    picked = picked.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    picked = " ".join(picked.split())
    return picked[:160]


def _extract_global_upper_assignments(code: str) -> dict[str, str]:
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


def _changed_upper_keys(before: str, after: str) -> list[str]:
    b = _extract_global_upper_assignments(before)
    a = _extract_global_upper_assignments(after)
    keys: list[str] = []
    for key in sorted(set(b) | set(a)):
        if b.get(key) != a.get(key):
            keys.append(key)
    return keys


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", " ", "-"}:
            out.append(ch)
    return " ".join("".join(out).split())


def _append_failure_direction(
    *,
    description: str,
    reason: str,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"items": []}
    if FAILURE_DIRECTIONS_FILE.exists():
        try:
            old = json.loads(FAILURE_DIRECTIONS_FILE.read_text())
            if isinstance(old, dict) and isinstance(old.get("items"), list):
                data = old
        except json.JSONDecodeError:
            pass

    norm_desc = _normalize_text(description)

    for item in data["items"]:
        if not isinstance(item, dict):
            continue
        old_desc = _normalize_text(str(item.get("description") or ""))
        old_reason = str(item.get("reason") or "").strip()
        if old_desc == norm_desc and old_reason == reason:
            item["count"] = int(item.get("count", 1)) + 1
            FAILURE_DIRECTIONS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )
            return

    data["items"].append(
        {
            "description": description,
            "reason": reason,
            "count": 1,
            "first_seen_ts": time.time(),
        }
    )
    FAILURE_DIRECTIONS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _recent_failure_direction_text(limit: int = 12) -> list[str]:
    if not FAILURE_DIRECTIONS_FILE.exists():
        return []
    try:
        obj = json.loads(FAILURE_DIRECTIONS_FILE.read_text())
    except json.JSONDecodeError:
        return []
    items = obj.get("items")
    if not isinstance(items, list):
        return []

    lines: list[str] = []
    for item in items[-limit:]:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if desc:
            lines.append(f"- {desc} ({reason or 'failed'})")
    return lines


def _next_candidate_id() -> str:
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for path in CANDIDATE_ROOT.glob("cand-*"):
        name = path.name
        if not name.startswith("cand-"):
            continue
        tail = name[5:]
        if tail.isdigit():
            max_n = max(max_n, int(tail))
    return f"cand-{max_n + 1:06d}"


def _gpu_snapshots() -> list[tuple[int, int, int]]:
    """返回 [(gpu_id, mem_used_mb, util_percent), ...]。"""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:
        return [(0, 0, 0)]

    out: list[tuple[int, int, int]] = []
    for raw in r.stdout.splitlines():
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) != 3:
            continue
        try:
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue

    return out if out else [(0, 0, 0)]


def _find_free_gpu_ids(
    *,
    occupied_gpu_ids: set[int],
    mem_idle_threshold_mb: int,
    util_idle_threshold_pct: int,
) -> list[int]:
    free: list[int] = []
    for gpu_id, mem_used, util in _gpu_snapshots():
        if gpu_id in occupied_gpu_ids:
            continue
        if mem_used <= mem_idle_threshold_mb and util <= util_idle_threshold_pct:
            free.append(gpu_id)
    return free


def _pick_least_loaded_gpu_id() -> int:
    """在没有满足空闲阈值时，选择负载最低的 GPU 兜底派发。"""
    snapshots = _gpu_snapshots()
    best_gpu = snapshots[0][0]
    best_score = (snapshots[0][1], snapshots[0][2])
    for gpu_id, mem_used, util in snapshots[1:]:
        score = (mem_used, util)
        if score < best_score:
            best_score = score
            best_gpu = gpu_id
    return best_gpu


def _copy_base_train_to_candidate(base_train: Path, candidate_dir: Path) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    target = candidate_dir / "train.py"
    shutil.copy2(base_train, target)
    return target


def _create_candidate(
    *,
    agent: ClaudeCodeAgent,
    base_train_path: Path,
    run_summaries: list[str],
    topic: str,
) -> CandidateTask | None:
    candidate_id = _next_candidate_id()
    candidate_dir = CANDIDATE_ROOT / candidate_id
    candidate_train = _copy_base_train_to_candidate(base_train_path, candidate_dir)
    base_code = candidate_train.read_text()

    failed_text = _recent_failure_direction_text()
    failed_block = "\n".join(failed_text) if failed_text else "无"

    forbidden_training_text = (
        "禁止运行任何训练命令或长任务命令，"
        "包括但不限于：uv run python train.py、python train.py、torchrun、nohup。"
        "训练只能由外层调度器执行。"
    )
    forbidden_git_history_text = (
        "禁止使用 git 命令查看、切换或回滚历史版本（如 git show/log/checkout/restore）。"
        "历史实验信息只能来自 autorunner/candidates 目录、results.tsv 与 artifacts 文件。"
    )
    eval_guidance = _build_eval_guidance()

    result = agent.refine(
        current_files={"train.py": base_code},
        run_summaries=run_summaries,
        metric_key="val_bpb",
        metric_direction="minimize",
        topic=topic,
        extra_hints=(
            "你需要产出一个新候选方向。"
            "必须显式避开以下已失败方向，不要重复同类尝试：\n"
            f"{failed_block}\n"
            "必须通过读取历史结果自行识别并规避重复方向；"
            "若方向近似，必须在 DESCRIPTION 中说明关键差异。\n"
            f"{eval_guidance}\n"
            f"{forbidden_training_text}\n"
            f"{forbidden_git_history_text}"
        ),
        workdir=candidate_dir,
    )

    (candidate_dir / "llm_refine.log").write_text(result.content or "")

    # 防止 agent 越权自行启动训练；若发现候选目录已有训练日志，直接废弃该候选。
    suspicious_logs = [
        candidate_dir / "run.log",
        candidate_dir / "run_repair_1.log",
        candidate_dir / "run_repair_2.log",
        candidate_dir / "run_repair_3.log",
    ]
    for p in suspicious_logs:
        if p.exists() and p.stat().st_size > 0:
            _append_queue_event(
                "candidate_generation_failed",
                {
                    "candidate_id": candidate_id,
                    "reason": "agent_attempted_training",
                    "path": str(p.relative_to(WORKDIR)),
                },
            )
            return None

    if not result.success:
        _append_queue_event(
            "candidate_generation_failed",
            {
                "candidate_id": candidate_id,
                "reason": (result.stderr or "generator_failed")[:200],
            },
        )
        return None

    new_code = result.files.get("train.py", "").strip()
    if not new_code:
        _append_queue_event(
            "candidate_generation_failed",
            {
                "candidate_id": candidate_id,
                "reason": "train.py not updated",
            },
        )
        return None

    candidate_train.write_text(new_code)

    desc = _extract_description_from_llm_reply(
        result.content,
        fallback=f"candidate {candidate_id}",
    )
    changed_keys = _changed_upper_keys(base_code, new_code)

    task = CandidateTask(
        candidate_id=candidate_id,
        workdir=candidate_dir,
        train_py_path=candidate_train,
        refine_description=desc,
        parent_ref=base_train_path.name,
        created_at=time.time(),
        queued_at=time.time(),
        status="queued",
    )

    (candidate_dir / "meta.json").write_text(
        json.dumps(asdict(task), indent=2, ensure_ascii=False, default=str)
    )

    _append_queue_event(
        "candidate_queued",
        {
            "candidate_id": candidate_id,
            "description": desc,
            "changed_upper_keys": changed_keys,
        },
    )
    return task


def _append_results_tsv(
    *,
    candidate_id: str,
    exp_result: ExperimentResult,
    decision: str,
    description: str,
):
    val_bpb = exp_result.val_bpb if exp_result.val_bpb is not None else 0.0
    peak_vram_mb = (
        exp_result.peak_vram_mb if exp_result.peak_vram_mb is not None else 0.0
    )
    memory_gb = peak_vram_mb / 1024.0
    status = "keep" if decision == "keep" else "discard"
    line = f"{candidate_id}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with RESULTS_TSV_FILE.open("a") as f:
        f.write(line)


def _run_one_candidate(
    *,
    task: CandidateTask,
    gpu_id: int,
    time_budget: int,
) -> ExperimentResult:
    def _runtime_env_for_gpu(gid: int) -> dict[str, str]:
        existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
        pythonpath = str(WORKDIR)
        if existing_pythonpath:
            pythonpath = f"{pythonpath}:{existing_pythonpath}"
        return {
            "CUDA_VISIBLE_DEVICES": str(gid),
            "PYTHONPATH": pythonpath,
        }

    result = run(
        train_py_path=task.train_py_path,
        time_budget=time_budget,
        run_log_path=task.workdir / "run.log",
        env_overrides=_runtime_env_for_gpu(gpu_id),
    )
    return result


def _repair_candidate_if_needed(
    *,
    agent: ClaudeCodeAgent,
    task: CandidateTask,
    initial_result: ExperimentResult,
    gpu_id: int,
    time_budget: int,
) -> tuple[ExperimentResult, bool, str, str, int]:
    """失败候选触发受约束 repair；返回 (最终结果, 是否修复成功, 最终描述)。"""
    final_desc = task.refine_description
    if initial_result.status == "completed":
        return initial_result, False, final_desc, "not_needed", 0

    max_repair_attempts = 3
    current_result = initial_result
    attempts_started = 0

    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    pythonpath = str(WORKDIR)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}:{existing_pythonpath}"

    for attempt in range(1, max_repair_attempts + 1):
        analysis = analyze_failure(
            current_result.failure_type or "runtime",
            current_result.log_tail,
        )
        if not analysis.should_retry:
            break
        attempts_started += 1

        _append_queue_event(
            "candidate_repair_start",
            {
                "candidate_id": task.candidate_id,
                "attempt": attempt,
                "failure_type": analysis.failure_type,
            },
        )

        current_code = task.train_py_path.read_text()
        issues = (
            f"failure_type: {analysis.failure_type}\n"
            f"repair_strategy: {analysis.repair_strategy}\n\n"
            "错误日志尾部（最多 50 行）：\n"
            f"{current_result.log_tail}"
        )
        repair_result = agent.repair(
            files={"train.py": current_code},
            issues=issues,
            refine_description=task.refine_description,
            workdir=task.workdir,
        )

        (task.workdir / f"llm_repair_{attempt}.log").write_text(
            repair_result.content or ""
        )

        if not repair_result.success:
            _append_queue_event(
                "candidate_repair_failed",
                {
                    "candidate_id": task.candidate_id,
                    "attempt": attempt,
                    "reason": (repair_result.stderr or "repair_failed")[:200],
                },
            )
            continue

        repaired_code = repair_result.files.get("train.py", "").strip()
        if not repaired_code or repaired_code == current_code:
            _append_queue_event(
                "candidate_repair_failed",
                {
                    "candidate_id": task.candidate_id,
                    "attempt": attempt,
                    "reason": "no_code_change",
                },
            )
            continue

        forbidden_keys = _changed_upper_keys(current_code, repaired_code)
        if forbidden_keys:
            _append_queue_event(
                "candidate_repair_out_of_scope",
                {
                    "candidate_id": task.candidate_id,
                    "attempt": attempt,
                    "changed_upper_keys": forbidden_keys[:12],
                },
            )
            _append_failure_direction(
                description=task.refine_description,
                reason="repair_out_of_scope",
            )
            return current_result, False, final_desc, "out_of_scope", attempts_started

        task.train_py_path.write_text(repaired_code)

        rerun_result = run(
            train_py_path=task.train_py_path,
            time_budget=time_budget,
            run_log_path=task.workdir / f"run_repair_{attempt}.log",
            env_overrides={
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "PYTHONPATH": pythonpath,
            },
        )

        candidate_desc = _extract_description_from_llm_reply(
            repair_result.content,
            fallback=task.refine_description,
        )
        low = candidate_desc.lower()
        if (
            "repair" in low
            or "bug" in low
            or "报错" in candidate_desc
            or "错误" in candidate_desc
            or "修复" in candidate_desc
            or "异常" in candidate_desc
        ):
            final_desc = task.refine_description
        else:
            final_desc = candidate_desc

        _append_queue_event(
            "candidate_repair_rerun_done",
            {
                "candidate_id": task.candidate_id,
                "attempt": attempt,
                "status": rerun_result.status,
                "val_bpb": rerun_result.val_bpb,
            },
        )

        if rerun_result.status == "completed":
            return rerun_result, True, final_desc, "success", attempts_started

        current_result = rerun_result

    if attempts_started > 0:
        return current_result, False, final_desc, "failed", attempts_started
    return current_result, False, final_desc, "not_needed", 0


def run_parallel_loop(
    *,
    max_total_runs: int,
    queue_capacity: int,
    low_watermark: int,
    max_running: int,
    time_budget: int,
    topic: str,
    mem_idle_threshold_mb: int,
    util_idle_threshold_pct: int,
    dashboard_status_filter: str,
    dispatch_stall_sec: float,
    poll_interval_sec: float,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

    if not BEST_TRAIN_FILE.exists():
        shutil.copy2(WORKDIR / "train.py", BEST_TRAIN_FILE)

    agent = ClaudeCodeAgent(model=MODEL, timeout_sec=600, stream_output=False)
    baseline_bpb = _load_baseline()

    queued: list[CandidateTask] = []
    running: dict[str, RunningTask] = {}
    completed_runs = 0
    executor = ThreadPoolExecutor(max_workers=max_running)
    failure_semantic_index = DescriptionEmbeddingIndex.from_failure_file(
        FAILURE_DIRECTIONS_FILE,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    )
    failure_reason_counts: dict[str, int] = {}
    repair_stats = {"start": 0, "success": 0, "failed": 0, "out_of_scope": 0}
    last_no_gpu_event_ts = 0.0
    finished_recent: list[dict[str, str]] = []
    start_ts = time.time()

    try:
        with Live(
            _render_dashboard(
                start_ts=start_ts,
                completed_runs=completed_runs,
                max_total_runs=max_total_runs,
                baseline_bpb=baseline_bpb,
                queued=queued,
                running=running,
                finished_recent=finished_recent,
                repair_stats=repair_stats,
                failure_reason_counts=failure_reason_counts,
                dashboard_status_filter=dashboard_status_filter,
                mem_idle_threshold_mb=mem_idle_threshold_mb,
                util_idle_threshold_pct=util_idle_threshold_pct,
            ),
            console=console,
            refresh_per_second=2,
            transient=False,
        ) as live:
            stop_refresh = threading.Event()

            def _snapshot_for_dashboard() -> tuple[
                int,
                float,
                list[CandidateTask],
                dict[str, RunningTask],
                list[dict[str, str]],
                dict[str, int],
                dict[str, int],
            ]:
                # 在无锁前提下做快照，遇到并发修改重试，避免看板线程崩溃。
                for _ in range(3):
                    try:
                        return (
                            completed_runs,
                            baseline_bpb,
                            list(queued),
                            dict(running),
                            list(finished_recent),
                            dict(repair_stats),
                            dict(failure_reason_counts),
                        )
                    except RuntimeError:
                        continue
                return (
                    completed_runs,
                    baseline_bpb,
                    [],
                    {},
                    [],
                    dict(repair_stats),
                    dict(failure_reason_counts),
                )

            def _refresh_loop() -> None:
                interval = max(0.5, min(1.0, poll_interval_sec))
                while not stop_refresh.is_set():
                    (
                        completed_snapshot,
                        baseline_snapshot,
                        queued_snapshot,
                        running_snapshot,
                        finished_snapshot,
                        repair_snapshot,
                        failure_snapshot,
                    ) = _snapshot_for_dashboard()
                    live.update(
                        _render_dashboard(
                            start_ts=start_ts,
                            completed_runs=completed_snapshot,
                            max_total_runs=max_total_runs,
                            baseline_bpb=baseline_snapshot,
                            queued=queued_snapshot,
                            running=running_snapshot,
                            finished_recent=finished_snapshot,
                            repair_stats=repair_snapshot,
                            failure_reason_counts=failure_snapshot,
                            dashboard_status_filter=dashboard_status_filter,
                            mem_idle_threshold_mb=mem_idle_threshold_mb,
                            util_idle_threshold_pct=util_idle_threshold_pct,
                        )
                    )
                    stop_refresh.wait(interval)

            refresh_thread = threading.Thread(target=_refresh_loop, daemon=True)
            refresh_thread.start()

            while True:
                # 1) 补充候选队列
                if len(queued) < low_watermark and completed_runs < max_total_runs:
                    need = min(
                        queue_capacity - len(queued), low_watermark - len(queued)
                    )
                    if need > 0:
                        run_summaries = _build_run_summaries()
                        for _ in range(need):
                            task = _create_candidate(
                                agent=agent,
                                base_train_path=BEST_TRAIN_FILE,
                                run_summaries=run_summaries,
                                topic=topic,
                            )
                            if task is not None:
                                queued.append(task)

                # 2) 空闲 GPU 启动实验
                occupied = {
                    rt.task.gpu_id
                    for rt in running.values()
                    if rt.task.gpu_id is not None
                }
                free_gpu_ids = _find_free_gpu_ids(
                    occupied_gpu_ids={int(x) for x in occupied},
                    mem_idle_threshold_mb=mem_idle_threshold_mb,
                    util_idle_threshold_pct=util_idle_threshold_pct,
                )

                # 队列存在但没有空闲 GPU 时，记录可观测事件，便于调参排障。
                if queued and not free_gpu_ids and len(running) < max_running:
                    now_ts = time.time()
                    if now_ts - last_no_gpu_event_ts >= 10.0:
                        oldest_wait = now_ts - min(x.queued_at for x in queued)
                        _append_queue_event(
                            "dispatch_waiting_gpu",
                            {
                                "queued": len(queued),
                                "running": len(running),
                                "oldest_wait_sec": round(oldest_wait, 1),
                                "mem_idle_threshold_mb": mem_idle_threshold_mb,
                                "util_idle_threshold_pct": util_idle_threshold_pct,
                            },
                        )
                        last_no_gpu_event_ts = now_ts

                # 若候选排队过久，使用最低负载 GPU 兜底派发，避免“已生成但不运行”。
                if (
                    queued
                    and not free_gpu_ids
                    and len(running) < max_running
                    and dispatch_stall_sec > 0
                ):
                    oldest_wait = time.time() - min(x.queued_at for x in queued)
                    if oldest_wait >= dispatch_stall_sec:
                        fallback_gpu = _pick_least_loaded_gpu_id()
                        free_gpu_ids = [fallback_gpu]
                        _append_queue_event(
                            "dispatch_fallback_gpu",
                            {
                                "gpu_id": fallback_gpu,
                                "oldest_wait_sec": round(oldest_wait, 1),
                            },
                        )

                while free_gpu_ids and queued and len(running) < max_running:
                    gpu_id = free_gpu_ids.pop(0)
                    task = queued.pop(0)
                    task.status = "running"
                    task.gpu_id = gpu_id
                    task.launch_ts = time.time()
                    fut = executor.submit(
                        _run_one_candidate,
                        task=task,
                        gpu_id=gpu_id,
                        time_budget=time_budget,
                    )
                    running[task.candidate_id] = RunningTask(task=task, future=fut)
                    _append_queue_event(
                        "candidate_running",
                        {
                            "candidate_id": task.candidate_id,
                            "gpu_id": gpu_id,
                        },
                    )

                # 3) 回收完成任务
                finished_ids: list[str] = []
                for cid, rt in running.items():
                    if not rt.future.done():
                        continue

                    exp_result = rt.future.result()
                    task = rt.task
                    finished_ids.append(cid)
                    completed_runs += 1

                    task.status = "repairing"
                    exp_result, repaired, final_desc, repair_status, repair_attempts = (
                        _repair_candidate_if_needed(
                            agent=agent,
                            task=task,
                            initial_result=exp_result,
                            gpu_id=task.gpu_id if task.gpu_id is not None else 0,
                            time_budget=time_budget,
                        )
                    )
                    task.status = "running"
                    task.final_description = final_desc
                    if repair_attempts > 0:
                        repair_stats["start"] += repair_attempts
                    if repair_status in {"success", "failed", "out_of_scope"}:
                        repair_stats[repair_status] += 1

                    if (
                        exp_result.status == "completed"
                        and exp_result.val_bpb is not None
                    ):
                        baseline_before = baseline_bpb
                        improved, decision = metric_judge(
                            baseline_bpb, exp_result.val_bpb
                        )
                    else:
                        baseline_before = baseline_bpb
                        improved, decision = False, "discard"

                    _append_results_tsv(
                        candidate_id=task.candidate_id,
                        exp_result=exp_result,
                        decision=decision,
                        description=task.final_description or task.refine_description,
                    )

                    discard_reason: str | None = None
                    if improved and exp_result.val_bpb is not None:
                        baseline_bpb = exp_result.val_bpb
                        shutil.copy2(task.train_py_path, BEST_TRAIN_FILE)
                        _append_queue_event(
                            "candidate_keep",
                            {
                                "candidate_id": task.candidate_id,
                                "val_bpb": exp_result.val_bpb,
                                "baseline": baseline_bpb,
                                "repaired": repaired,
                            },
                        )
                        final_status = "keep"
                    else:
                        discard_reason = (
                            "repair_out_of_scope"
                            if repair_status == "out_of_scope"
                            else (
                                exp_result.failure_type
                                if exp_result.status != "completed"
                                else "metric_not_improved"
                            )
                        )
                        _append_failure_direction(
                            description=task.refine_description,
                            reason=discard_reason,
                        )
                        failure_semantic_index.add_item(
                            {
                                "description": task.refine_description,
                                "reason": discard_reason,
                            }
                        )
                        failure_reason_counts[discard_reason] = (
                            failure_reason_counts.get(discard_reason, 0) + 1
                        )
                        _append_queue_event(
                            "candidate_discard",
                            {
                                "candidate_id": task.candidate_id,
                                "status": exp_result.status,
                                "failure_type": exp_result.failure_type,
                                "repaired": repaired,
                                "discard_reason": discard_reason,
                            },
                        )
                        final_status = "discard"

                    eval_path = _write_parallel_eval_artifact(
                        task=task,
                        exp_result=exp_result,
                        baseline_before=baseline_before,
                        baseline_after=baseline_bpb,
                        improved=improved,
                        decision=decision,
                        repaired=repaired,
                        discard_reason=discard_reason,
                    )
                    _append_parallel_current_state(
                        task=task,
                        exp_result=exp_result,
                        decision=decision,
                        improved=improved,
                        baseline_before=baseline_before,
                        baseline_after=baseline_bpb,
                        eval_path=eval_path,
                    )

                    runtime_s = int(
                        max(0, time.time() - (task.launch_ts or time.time()))
                    )
                    wait_s = int(
                        max(0, (task.launch_ts or time.time()) - task.queued_at)
                    )
                    finished_recent.append(
                        {
                            "candidate_id": task.candidate_id,
                            "status": final_status,
                            "gpu": str(task.gpu_id) if task.gpu_id is not None else "-",
                            "wait": str(wait_s),
                            "runtime": str(runtime_s),
                            "desc": task.final_description or task.refine_description,
                        }
                    )
                    if len(finished_recent) > 30:
                        finished_recent = finished_recent[-30:]

                for cid in finished_ids:
                    running.pop(cid, None)

                # 4) 终止条件
                if completed_runs >= max_total_runs and not running:
                    break

                time.sleep(poll_interval_sec)

            stop_refresh.set()
            refresh_thread.join(timeout=1.0)
    finally:
        executor.shutdown(wait=True)

    console.print(
        f"[parallel_runner] finished completed_runs={completed_runs} "
        f"best_val_bpb={baseline_bpb if baseline_bpb != float('inf') else 'inf'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Parallel Runner")
    parser.add_argument("--max-total-runs", type=int, default=10)
    parser.add_argument("--queue-capacity", type=int, default=6)
    parser.add_argument("--low-watermark", type=int, default=2)
    parser.add_argument("--max-running", type=int, default=1)
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--mem-idle-threshold-mb", type=int, default=20000)
    parser.add_argument("--util-idle-threshold-pct", type=int, default=85)
    parser.add_argument(
        "--dashboard-status-filter",
        type=str,
        choices=["all", "active", "failed", "finished"],
        default="all",
    )
    parser.add_argument("--dispatch-stall-sec", type=float, default=45.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--topic",
        type=str,
        default=(
            "你是 AI 研究员，正在对 GPT 模型训练代码 train.py 进行实验优化。"
            "目标：降低 val_bpb（validation bits per byte）。"
            "val_bpb 越低越好。每轮请提出一个具体的改进思路并直接修改 train.py。"
        ),
    )
    args = parser.parse_args()

    run_parallel_loop(
        max_total_runs=args.max_total_runs,
        queue_capacity=args.queue_capacity,
        low_watermark=args.low_watermark,
        max_running=args.max_running,
        time_budget=args.time_budget,
        topic=args.topic,
        mem_idle_threshold_mb=args.mem_idle_threshold_mb,
        util_idle_threshold_pct=args.util_idle_threshold_pct,
        dashboard_status_filter=args.dashboard_status_filter,
        dispatch_stall_sec=args.dispatch_stall_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
