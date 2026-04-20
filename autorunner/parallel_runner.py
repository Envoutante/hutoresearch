"""parallel_runner: 并行候选生成与实验调度（第一版，单文件实现）"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# 支持直接运行: python autorunner/parallel_runner.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.claude_code_agent import ClaudeCodeAgent
from autorunner.experiment_executor import (
    ExperimentResult,
    analyze_failure,
    judge as metric_judge,
    run,
)


WORKDIR = Path("/mount/disk1/rl-hyr/autoresearch")
ARTIFACTS_DIR = WORKDIR / "autorunner" / "artifacts"
CANDIDATE_ROOT = WORKDIR / "autorunner" / "candidates"
RESULTS_TSV_FILE = WORKDIR / "results.tsv"
FAILURE_DIRECTIONS_FILE = ARTIFACTS_DIR / "failure_directions.json"
QUEUE_EVENTS_FILE = ARTIFACTS_DIR / "parallel_queue.jsonl"
BEST_TRAIN_FILE = ARTIFACTS_DIR / "best_train.py"
MODEL = "MiniMax-M2.7"


@dataclass
class CandidateTask:
    candidate_id: str
    workdir: Path
    train_py_path: Path
    refine_description: str
    direction_fingerprint: str
    parent_ref: str
    created_at: float
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


def _build_direction_fingerprint(description: str, changed_keys: list[str]) -> str:
    payload = {
        "desc": _normalize_text(description),
        "keys": sorted(changed_keys),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_failure_direction_set() -> set[str]:
    if not FAILURE_DIRECTIONS_FILE.exists():
        return set()
    try:
        obj = json.loads(FAILURE_DIRECTIONS_FILE.read_text())
    except json.JSONDecodeError:
        return set()

    items = obj.get("items")
    if not isinstance(items, list):
        return set()

    out: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        fp = str(it.get("fingerprint") or "").strip()
        if fp:
            out.add(fp)
    return out


def _append_failure_direction(
    *,
    fingerprint: str,
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

    for item in data["items"]:
        if item.get("fingerprint") == fingerprint:
            item["count"] = int(item.get("count", 1)) + 1
            FAILURE_DIRECTIONS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )
            return

    data["items"].append(
        {
            "fingerprint": fingerprint,
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
    failure_fingerprints: set[str],
    topic: str,
) -> CandidateTask | None:
    candidate_id = _next_candidate_id()
    candidate_dir = CANDIDATE_ROOT / candidate_id
    candidate_train = _copy_base_train_to_candidate(base_train_path, candidate_dir)
    base_code = candidate_train.read_text()

    failed_text = _recent_failure_direction_text()
    failed_block = "\n".join(failed_text) if failed_text else "无"

    result = agent.refine(
        current_files={"train.py": base_code},
        run_summaries=run_summaries,
        metric_key="val_bpb",
        metric_direction="minimize",
        topic=topic,
        extra_hints=(
            "你需要产出一个新候选方向。"
            "必须显式避开以下已失败方向，不要重复同类尝试：\n"
            f"{failed_block}"
        ),
        workdir=candidate_dir,
    )

    (candidate_dir / "llm_refine.log").write_text(result.content or "")

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
    fingerprint = _build_direction_fingerprint(desc, changed_keys)

    if fingerprint in failure_fingerprints:
        _append_queue_event(
            "candidate_rejected",
            {
                "candidate_id": candidate_id,
                "fingerprint": fingerprint,
                "reason": "duplicate_failed_direction",
            },
        )
        _append_failure_direction(
            fingerprint=fingerprint,
            description=desc,
            reason="duplicate_failed_direction",
        )
        return None

    task = CandidateTask(
        candidate_id=candidate_id,
        workdir=candidate_dir,
        train_py_path=candidate_train,
        refine_description=desc,
        direction_fingerprint=fingerprint,
        parent_ref=base_train_path.name,
        created_at=time.time(),
        status="queued",
    )

    (candidate_dir / "meta.json").write_text(
        json.dumps(asdict(task), indent=2, ensure_ascii=False, default=str)
    )

    _append_queue_event(
        "candidate_queued",
        {
            "candidate_id": candidate_id,
            "fingerprint": fingerprint,
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
    print(f"[runner] launch {task.candidate_id} on gpu={gpu_id}")
    result = run(
        train_py_path=task.train_py_path,
        time_budget=time_budget,
        run_log_path=task.workdir / "run.log",
        env_overrides={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    return result


def _repair_candidate_if_needed(
    *,
    agent: ClaudeCodeAgent,
    task: CandidateTask,
    initial_result: ExperimentResult,
    gpu_id: int,
    time_budget: int,
) -> tuple[ExperimentResult, bool, str]:
    """失败候选触发受约束 repair；返回 (最终结果, 是否修复成功, 最终描述)。"""
    final_desc = task.refine_description
    if initial_result.status == "completed":
        return initial_result, False, final_desc

    max_repair_attempts = 3
    current_result = initial_result

    for attempt in range(1, max_repair_attempts + 1):
        analysis = analyze_failure(
            current_result.failure_type or "runtime",
            current_result.log_tail,
        )
        if not analysis.should_retry:
            break

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
                fingerprint=task.direction_fingerprint,
                description=task.refine_description,
                reason="repair_out_of_scope",
            )
            return current_result, False, final_desc

        task.train_py_path.write_text(repaired_code)

        rerun_result = run(
            train_py_path=task.train_py_path,
            time_budget=time_budget,
            run_log_path=task.workdir / f"run_repair_{attempt}.log",
            env_overrides={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
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
            return rerun_result, True, final_desc

        current_result = rerun_result

    return current_result, False, final_desc


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
    poll_interval_sec: float,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

    if not BEST_TRAIN_FILE.exists():
        shutil.copy2(WORKDIR / "train.py", BEST_TRAIN_FILE)

    agent = ClaudeCodeAgent(model=MODEL, timeout_sec=600)
    baseline_bpb = _load_baseline()

    queued: list[CandidateTask] = []
    running: dict[str, RunningTask] = {}
    completed_runs = 0
    executor = ThreadPoolExecutor(max_workers=max_running)
    failure_fingerprints = _load_failure_direction_set()

    print("[parallel_runner] started")
    print(
        f"[parallel_runner] baseline={baseline_bpb if baseline_bpb != float('inf') else 'inf'} "
        f"queue_capacity={queue_capacity} max_running={max_running}"
    )

    try:
        while True:
            # 1) 补充候选队列
            if len(queued) < low_watermark and completed_runs < max_total_runs:
                need = min(queue_capacity - len(queued), low_watermark - len(queued))
                if need > 0:
                    run_summaries = _build_run_summaries()
                    for _ in range(need):
                        task = _create_candidate(
                            agent=agent,
                            base_train_path=BEST_TRAIN_FILE,
                            run_summaries=run_summaries,
                            failure_fingerprints=failure_fingerprints,
                            topic=topic,
                        )
                        if task is not None:
                            queued.append(task)

            # 2) 空闲 GPU 启动实验
            occupied = {
                rt.task.gpu_id for rt in running.values() if rt.task.gpu_id is not None
            }
            free_gpu_ids = _find_free_gpu_ids(
                occupied_gpu_ids={int(x) for x in occupied},
                mem_idle_threshold_mb=mem_idle_threshold_mb,
                util_idle_threshold_pct=util_idle_threshold_pct,
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

                exp_result, repaired, final_desc = _repair_candidate_if_needed(
                    agent=agent,
                    task=task,
                    initial_result=exp_result,
                    gpu_id=task.gpu_id if task.gpu_id is not None else 0,
                    time_budget=time_budget,
                )
                task.final_description = final_desc

                if exp_result.status == "completed" and exp_result.val_bpb is not None:
                    improved, decision = metric_judge(baseline_bpb, exp_result.val_bpb)
                else:
                    improved, decision = False, "discard"

                _append_results_tsv(
                    candidate_id=task.candidate_id,
                    exp_result=exp_result,
                    decision=decision,
                    description=task.final_description or task.refine_description,
                )

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
                else:
                    _append_failure_direction(
                        fingerprint=task.direction_fingerprint,
                        description=task.refine_description,
                        reason=(
                            exp_result.failure_type
                            if exp_result.status != "completed"
                            else "metric_not_improved"
                        ),
                    )
                    failure_fingerprints.add(task.direction_fingerprint)
                    _append_queue_event(
                        "candidate_discard",
                        {
                            "candidate_id": task.candidate_id,
                            "status": exp_result.status,
                            "failure_type": exp_result.failure_type,
                            "repaired": repaired,
                        },
                    )

                print(
                    f"[runner] done {task.candidate_id} gpu={task.gpu_id} "
                    f"status={exp_result.status} decision={decision} "
                    f"val_bpb={exp_result.val_bpb}"
                )

            for cid in finished_ids:
                running.pop(cid, None)

            # 4) 终止条件
            if completed_runs >= max_total_runs and not running:
                break

            time.sleep(poll_interval_sec)
    finally:
        executor.shutdown(wait=True)

    print(
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
    parser.add_argument("--mem-idle-threshold-mb", type=int, default=800)
    parser.add_argument("--util-idle-threshold-pct", type=int, default=10)
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
        poll_interval_sec=args.poll_interval_sec,
    )
