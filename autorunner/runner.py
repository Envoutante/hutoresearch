"""runner: 并行候选生成与实验调度。"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import py_compile
import re
import shutil
import smtplib
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# 支持直接运行: python autorunner/runner.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.code_agent import BaseCodeAgent, make_code_agent
from autorunner.env_config import load_dotenv, project_root
from autorunner.experiment_executor import (
    ExperimentResult,
    analyze_failure,
    judge as metric_judge,
    run,
)
from autorunner.search_policy import (
    SearchPlan,
    complete_plan_from_payload,
    compute_reward,
    plan_prompt_payload,
    search_plan_to_prompt,
    select_search_plan,
    validate_search_plan,
)

PROJECT_ROOT = project_root()
load_dotenv()

WORKDIR = Path(os.getenv("AR_WORKDIR", str(PROJECT_ROOT))).expanduser()
ARTIFACTS_DIR = WORKDIR / "autorunner" / "artifacts"
CANDIDATE_ROOT = WORKDIR / "autorunner" / "candidates"
RESULTS_TSV_FILE = WORKDIR / "results.tsv"
STATE_DIR = ARTIFACTS_DIR / "state"
GUIDANCE_DIR = ARTIFACTS_DIR / "guidance"
SNAPSHOTS_DIR = ARTIFACTS_DIR / "snapshots"
ALERTS_DIR = ARTIFACTS_DIR / "alerts"
FAILURE_DIRECTIONS_FILE = STATE_DIR / "failure_directions.json"
EXPERIMENT_REGISTRY_FILE = STATE_DIR / "experiment_registry.jsonl"
GENERATOR_GUIDANCE_FILE = GUIDANCE_DIR / "generator_guidance.md"
QUEUE_EVENTS_FILE = STATE_DIR / "parallel_queue.jsonl"
BEST_TRAIN_FILE = SNAPSHOTS_DIR / "best_train.py"
BEST_SNAPSHOT_FILE = STATE_DIR / "best_snapshot.json"
MODEL = os.getenv("AR_MODEL", "deepseek-v4-pro[1m]")
AGENT_BACKEND = os.getenv("AR_AGENT_BACKEND", "claude")
console = Console()
REGISTRY_LOCK = threading.Lock()
GENERATOR_GUIDANCE_LOCK = threading.Lock()
BEST_TRAIN_LOCK = threading.Lock()
ACTIVE_REGISTRY_STALE_SEC = int(os.getenv("AR_REGISTRY_ACTIVE_STALE_SEC", "1800"))

FATAL_STOP_REASON = "fatal_model_api_error"


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
    direction_key: str = ""
    hypothesis: str = ""
    mechanism: str = ""
    touched_symbols: list[str] | None = None
    novelty_claim: str = ""
    changed_upper_keys: list[str] | None = None
    search_plan_id: str = ""
    search_mode: str = ""
    search_operator: str = ""
    search_parent_id: str = ""
    semantic_parent_id: str = ""
    code_parent_id: str = ""
    code_parent_ref: str = ""
    code_parent_hash: str = ""
    relation_edges: list[dict[str, Any]] | None = None
    target_direction_key: str = ""
    intent: str = ""
    rationale: str = ""
    avoid_direction_keys: list[str] | None = None


@dataclass
class RunningTask:
    task: CandidateTask
    future: Future[ExperimentResult]


class FatalModelAPIError(RuntimeError):
    """模型 API 出现致命错误时，用于触发全局熔断。"""


def _has_fatal_model_error(*texts: str) -> bool:
    merged = "\n".join(str(x or "") for x in texts if x).strip()
    if not merged:
        return False

    lower = merged.lower()
    for line in lower.splitlines():
        if "status" not in line and "http" not in line and "error" not in line:
            continue
        if re.search(r"\b(?:4\d{2}|5\d{2})\b", line):
            return True
    return False


def _raise_if_fatal_model_error(*texts: str) -> None:
    if _has_fatal_model_error(*texts):
        raise FatalModelAPIError(FATAL_STOP_REASON)


def _send_qq_alert_email(subject: str, body: str) -> tuple[bool, str]:
    enabled_raw = os.getenv("AR_ALERT_ENABLE", "1").strip().lower()
    if enabled_raw in {"0", "false", "no", "off"}:
        return False, "alert_disabled"

    host = os.getenv("AR_ALERT_SMTP_HOST", "smtp.qq.com").strip()
    port_text = os.getenv("AR_ALERT_SMTP_PORT", "465").strip()
    user = os.getenv("AR_ALERT_SMTP_USER", "").strip()
    password = os.getenv("AR_ALERT_SMTP_PASS", "").strip()
    sender = os.getenv("AR_ALERT_FROM", user).strip()
    recipient = os.getenv("AR_ALERT_TO", user).strip()
    use_ssl = os.getenv("AR_ALERT_USE_SSL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    use_starttls = os.getenv("AR_ALERT_STARTTLS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    if not host:
        return False, "smtp_host_missing"
    if not user or not password or not recipient:
        return False, "smtp_credentials_or_recipient_missing"

    try:
        port = int(port_text)
    except ValueError:
        return False, f"invalid_smtp_port={port_text}"

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.ehlo()
                if use_starttls:
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(user, password)
                smtp.send_message(msg)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    return True, "sent"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _append_queue_event(event_type: str, payload: dict):
    QUEUE_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now_iso(),
        "event": event_type,
        "payload": payload,
    }
    with QUEUE_EVENTS_FILE.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _load_experiment_registry() -> list[dict[str, Any]]:
    if not EXPERIMENT_REGISTRY_FILE.exists():
        return []
    return _read_experiment_registry_unlocked()


def _read_experiment_registry_unlocked() -> list[dict[str, Any]]:
    if not EXPERIMENT_REGISTRY_FILE.exists():
        return []

    items: list[dict[str, Any]] = []
    try:
        with EXPERIMENT_REGISTRY_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
    except OSError:
        return []
    return items


def _update_registry_entry(candidate_id: str, **updates: Any) -> None:
    if not candidate_id:
        return
    status = str(updates.get("status") or "").strip()
    if "reward" not in updates:
        if status == "blocked_duplicate":
            updates["reward"] = -0.4
            updates["reward_source"] = "blocked_duplicate_default"
        elif status == "generation_failed":
            updates["reward"] = -0.6
            updates["reward_source"] = "generation_failed_default"
        elif status == "aborted":
            updates["reward"] = -1.0
            updates["reward_source"] = "aborted_default"
    EXPERIMENT_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_LOCK:
        items = _read_experiment_registry_unlocked()
        now = _now_iso()
        found = False
        for item in items:
            if str(item.get("candidate_id") or "") != candidate_id:
                continue
            item.update(updates)
            item["updated_at"] = now
            found = True
        if not found:
            item = {"candidate_id": candidate_id, "created_at": now}
            item.update(updates)
            item["updated_at"] = now
            items.append(item)
        tmp_path = EXPERIMENT_REGISTRY_FILE.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(
                    json.dumps(item, ensure_ascii=False, default=_json_default) + "\n"
                )
        tmp_path.replace(EXPERIMENT_REGISTRY_FILE)


def _parse_registry_time(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _mark_stale_active_registry_entries(
    *,
    stale_after_sec: int = ACTIVE_REGISTRY_STALE_SEC,
) -> int:
    if stale_after_sec <= 0 or not EXPERIMENT_REGISTRY_FILE.exists():
        return 0

    active_statuses = {"proposed", "queued", "running"}
    now_ts = time.time()
    now_iso = _now_iso()
    stale_ids: list[str] = []

    with REGISTRY_LOCK:
        items = _read_experiment_registry_unlocked()
        changed = False
        for item in items:
            status = str(item.get("status") or "")
            if status not in active_statuses:
                continue
            ts = _parse_registry_time(item.get("updated_at") or item.get("created_at"))
            if ts is None or now_ts - ts < stale_after_sec:
                continue
            cid = str(item.get("candidate_id") or "")
            if not cid:
                continue
            item["status"] = "generation_failed"
            item["previous_status"] = status
            item["failure_reason"] = "stale_active_on_runner_start"
            item["updated_at"] = now_iso
            stale_ids.append(cid)
            changed = True

        if changed:
            tmp_path = EXPERIMENT_REGISTRY_FILE.with_suffix(".jsonl.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for item in items:
                    f.write(
                        json.dumps(item, ensure_ascii=False, default=_json_default)
                        + "\n"
                    )
            tmp_path.replace(EXPERIMENT_REGISTRY_FILE)

    if stale_ids:
        _append_queue_event(
            "registry_stale_active_marked",
            {
                "count": len(stale_ids),
                "candidate_ids": stale_ids[:20],
                "stale_after_sec": stale_after_sec,
            },
        )
    return len(stale_ids)


def _short_text(text: str, max_len: int = 42) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _split_symbol_list(text: str) -> list[str]:
    cleaned = str(text or "").replace("，", ",").replace("、", ",")
    out: list[str] = []
    for part in cleaned.split(","):
        item = part.strip().strip("[]")
        if item:
            out.append(item[:80])
    return out[:16]


def _extract_structured_refine_fields(reply: str, fallback_desc: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    wanted = {
        "DIRECTION_KEY",
        "HYPOTHESIS",
        "MECHANISM",
        "TOUCHED_SYMBOLS",
        "NOVELTY_CLAIM",
        "DESCRIPTION",
    }
    for raw in str(reply or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in wanted:
            fields[key] = (
                value.strip()
                if key == "DESCRIPTION"
                else " ".join(value.strip().split())
            )

    desc = fields.get("DESCRIPTION") or fallback_desc
    direction_key = fields.get("DIRECTION_KEY") or "unknown"
    return {
        "direction_key": direction_key[:120],
        "hypothesis": (fields.get("HYPOTHESIS") or "")[:500],
        "mechanism": (fields.get("MECHANISM") or "")[:500],
        "touched_symbols": _split_symbol_list(fields.get("TOUCHED_SYMBOLS") or ""),
        "novelty_claim": (fields.get("NOVELTY_CLAIM") or "")[:500],
        "description": desc,
    }


def _code_diff_summary(before: str, after: str, *, max_lines: int = 160) -> str:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="best_train.py",
        tofile="candidate_train.py",
        lineterm="",
        n=3,
    )
    lines = list(diff)
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-max(1, max_lines // 2) :]
        omitted = len(lines) - len(head) - len(tail)
        lines = head + [f"... omitted {omitted} diff lines ..."] + tail
    return "\n".join(lines)


def _registry_card(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "status": item.get("status"),
        "direction_key": item.get("direction_key"),
        "hypothesis": item.get("hypothesis"),
        "mechanism": item.get("mechanism"),
        "touched_symbols": item.get("touched_symbols"),
        "changed_upper_keys": item.get("changed_upper_keys"),
        "description": item.get("description"),
        "novelty_claim": item.get("novelty_claim"),
        "result": item.get("result"),
    }


def _registry_context_for_agent(
    *,
    registry_items: list[dict[str, Any]],
    exclude_candidate_id: str = "",
    limit_active: int = 12,
    limit_recent: int = 18,
) -> str:
    active_statuses = {"proposed", "queued", "running"}
    active = [
        _registry_card(x)
        for x in registry_items
        if str(x.get("status") or "") in active_statuses
        and str(x.get("candidate_id") or "") != exclude_candidate_id
    ][-limit_active:]
    recent = [
        _registry_card(x)
        for x in registry_items
        if str(x.get("status") or "") in {"keep", "discard", "blocked_duplicate"}
        and str(x.get("candidate_id") or "") != exclude_candidate_id
    ][-limit_recent:]
    payload = {
        "active_directions": active,
        "recent_finished_or_blocked": recent,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _result_dict(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result")
    return result if isinstance(result, dict) else {}


def _result_val_bpb(item: dict[str, Any]) -> float | None:
    result = _result_dict(item)
    value = result.get("val_bpb")
    try:
        bpb = float(value)
    except (TypeError, ValueError):
        return None
    return bpb if math.isfinite(bpb) else None


def _candidate_guidance_line(item: dict[str, Any], *, best_bpb: float | None) -> str:
    cid = str(item.get("candidate_id") or "")
    status = str(item.get("status") or "")
    direction = str(item.get("direction_key") or "")
    desc = str(item.get("description") or "").strip()
    result = _result_dict(item)
    val_bpb = _result_val_bpb(item)
    pieces = [cid, status]
    if direction:
        pieces.append(direction)
    if val_bpb is not None:
        metric = f"val_bpb={val_bpb:.6f}"
        if best_bpb is not None and best_bpb > 0:
            delta = _safe_relative_change_percent(best_bpb, val_bpb)
            if delta is not None:
                metric += f" delta_vs_best={delta:+.2f}%"
        pieces.append(metric)
    discard_reason = str(result.get("discard_reason") or "").strip()
    if discard_reason:
        pieces.append(f"reason={discard_reason}")
    if desc:
        pieces.append(desc)
    return "- " + " | ".join(pieces)


def _build_generator_guidance_text(
    *,
    registry_items: list[dict[str, Any]],
    exclude_candidate_id: str = "",
    max_recent: int = 14,
    max_active: int = 12,
    max_failures: int = 12,
) -> str:
    if exclude_candidate_id:
        registry_items = [
            item
            for item in registry_items
            if str(item.get("candidate_id") or "") != exclude_candidate_id
        ]

    completed = [
        item
        for item in registry_items
        if str(item.get("status") or "") in {"keep", "discard"}
        and _result_val_bpb(item) is not None
    ]
    best_item = min(
        completed,
        key=lambda x: _result_val_bpb(x) or float("inf"),
        default=None,
    )
    best_bpb = _result_val_bpb(best_item) if best_item else None

    active = [
        item
        for item in registry_items
        if str(item.get("status") or "") in {"proposed", "queued", "running"}
    ][-max_active:]
    recent = [
        item
        for item in registry_items
        if str(item.get("status") or "") in {"keep", "discard", "blocked_duplicate"}
    ][-max_recent:]
    blocked = [
        item
        for item in registry_items
        if str(item.get("status") or "") == "blocked_duplicate"
    ][-max_failures:]
    explicit_failures = _recent_failure_direction_text(limit=max_failures)

    lines = [
        "# Generator Guidance",
        f"updated_at: {_now_iso()}",
        "",
        (
            "Use this runner-provided guidance as the authoritative advice source. "
            "Do not read results.tsv, autorunner/artifacts, autorunner/candidates, "
            "current_state.md, eval-*.json, or git history."
        ),
        "",
        "## Current Best",
    ]
    if best_item is None or best_bpb is None:
        lines.append("- none yet")
    else:
        lines.append(_candidate_guidance_line(best_item, best_bpb=best_bpb))

    lines += ["", "## Active Or Reserved Directions"]
    if active:
        for item in active:
            lines.append(_candidate_guidance_line(item, best_bpb=best_bpb))
    else:
        lines.append("- none")

    lines += ["", "## Recent Finished Or Blocked Directions"]
    if recent:
        for item in recent:
            lines.append(_candidate_guidance_line(item, best_bpb=best_bpb))
    else:
        lines.append("- none")

    lines += ["", "## Avoid Or Pivot Away From"]
    if explicit_failures:
        lines.extend(explicit_failures)
    else:
        lines.append("- no metric failures recorded yet")
    for item in blocked:
        judge = (
            item.get("novelty_judge")
            if isinstance(item.get("novelty_judge"), dict)
            else {}
        )
        required_pivot = str(judge.get("required_pivot") or "").strip()
        reason = str(judge.get("reason") or "").strip()
        desc = str(item.get("description") or "").strip()
        detail = required_pivot or reason or "blocked by novelty judge"
        if desc:
            lines.append(f"- {desc} (blocked_duplicate: {detail})")

    lines += [
        "",
        "## Selection Rules",
        (
            "- Prefer mechanisms that are not already active, recently failed, "
            "or blocked as duplicates."
        ),
        (
            "- If using the same broad family as a recent candidate, change a different "
            "concrete mechanism and state that difference in NOVELTY_CLAIM."
        ),
        (
            "- Preserve known best behavior unless the proposal explicitly tests a "
            "different mechanism with a plausible benefit."
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _refresh_generator_guidance(*, exclude_candidate_id: str = "") -> str:
    GENERATOR_GUIDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATOR_GUIDANCE_LOCK:
        text = _build_generator_guidance_text(
            registry_items=_load_experiment_registry(),
            exclude_candidate_id=exclude_candidate_id,
        )
        tmp_path = GENERATOR_GUIDANCE_FILE.with_suffix(".md.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(GENERATOR_GUIDANCE_FILE)
        return text


def _select_judge_context(
    *,
    candidate: dict[str, Any],
    registry_items: list[dict[str, Any]],
    max_items: int = 30,
) -> list[dict[str, Any]]:
    direction_key = str(candidate.get("direction_key") or "").strip()
    touched = {
        str(x).strip().lower()
        for x in candidate.get("touched_symbols") or []
        if str(x).strip()
    }
    changed = {
        str(x).strip().lower()
        for x in candidate.get("changed_upper_keys") or []
        if str(x).strip()
    }
    active_statuses = {"proposed", "queued", "running"}

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for order, item in enumerate(registry_items):
        cid = str(item.get("candidate_id") or "")
        if cid == str(candidate.get("candidate_id") or ""):
            continue
        score = 0
        status = str(item.get("status") or "")
        if status in active_statuses:
            score += 100
        if direction_key and direction_key == str(item.get("direction_key") or ""):
            score += 80
        item_touched = {
            str(x).strip().lower()
            for x in item.get("touched_symbols") or []
            if str(x).strip()
        }
        item_changed = {
            str(x).strip().lower()
            for x in item.get("changed_upper_keys") or []
            if str(x).strip()
        }
        score += 12 * len(touched & item_touched)
        score += 10 * len(changed & item_changed)
        if score > 0:
            scored.append((score, order, _registry_card(item)))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cards = [x[2] for x in scored[:max_items]]
    if len(cards) < max_items:
        seen = {str(x.get("candidate_id") or "") for x in cards}
        for item in reversed(registry_items):
            cid = str(item.get("candidate_id") or "")
            if (
                not cid
                or cid in seen
                or cid == str(candidate.get("candidate_id") or "")
            ):
                continue
            cards.append(_registry_card(item))
            seen.add(cid)
            if len(cards) >= max_items:
                break
    return cards


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _candidate_py_compile_ok(train_py_path: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(train_py_path), doraise=True)
    except py_compile.PyCompileError as exc:
        return False, str(exc)[:500]
    except OSError as exc:
        return False, str(exc)[:500]
    return True, ""


def _structured_fields_are_usable(fields: dict[str, Any]) -> bool:
    direction_key = str(fields.get("direction_key") or "").strip()
    description = str(fields.get("description") or "").strip()
    mechanism = str(fields.get("mechanism") or "").strip()
    if not direction_key or direction_key == "unknown":
        return False
    if not description:
        return False
    if not mechanism:
        return False
    return True


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


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_best_snapshot_unlocked() -> dict[str, Any]:
    if not BEST_SNAPSHOT_FILE.exists():
        return {}
    try:
        data = json.loads(BEST_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_best_snapshot_unlocked(
    *,
    best_candidate_id: str,
    best_val_bpb: float | None,
    source: str,
) -> dict[str, Any]:
    BEST_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "best_candidate_id": best_candidate_id or "baseline",
        "best_val_bpb": best_val_bpb if best_val_bpb is not None else None,
        "best_train_path": str(BEST_TRAIN_FILE.relative_to(WORKDIR)),
        "code_hash": _file_sha256(BEST_TRAIN_FILE),
        "updated_at": _now_iso(),
        "source": source,
    }
    tmp_path = BEST_SNAPSHOT_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(BEST_SNAPSHOT_FILE)
    return payload


def _infer_best_snapshot_from_registry(
    baseline_bpb: float,
) -> tuple[str, float | None, str]:
    best_candidate_id = "baseline"
    best_val_bpb = baseline_bpb if math.isfinite(baseline_bpb) else None
    source = "baseline_init"
    for item in _load_experiment_registry():
        if str(item.get("status") or "") != "keep":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        try:
            val_bpb = float(result.get("val_bpb"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val_bpb):
            continue
        if best_val_bpb is None or val_bpb <= best_val_bpb:
            best_candidate_id = str(item.get("candidate_id") or "baseline")
            best_val_bpb = val_bpb
            source = "registry_inferred"
    return best_candidate_id, best_val_bpb, source


def _initialize_best_snapshot(baseline_bpb: float) -> None:
    with BEST_TRAIN_LOCK:
        if not BEST_TRAIN_FILE.exists():
            BEST_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WORKDIR / "train.py", BEST_TRAIN_FILE)
        snapshot = _read_best_snapshot_unlocked()
        if snapshot.get("best_candidate_id") and snapshot.get("code_hash"):
            return
        best_candidate_id, best_val_bpb, source = _infer_best_snapshot_from_registry(
            baseline_bpb
        )
        _write_best_snapshot_unlocked(
            best_candidate_id=best_candidate_id,
            best_val_bpb=best_val_bpb,
            source=source,
        )


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


def _safe_relative_change_percent(
    baseline_before: float,
    val_bpb: float | None,
) -> float | None:
    if val_bpb is None:
        return None
    if not math.isfinite(baseline_before) or baseline_before == 0:
        return None
    return (val_bpb - baseline_before) / baseline_before * 100.0


def _extract_description_from_llm_reply(reply: str, fallback: str) -> str:
    if not reply:
        return fallback

    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("DESCRIPTION:"):
            return line.split(":", 1)[1].strip() or fallback

    return fallback


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
    FAILURE_DIRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
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


def _next_candidate_number() -> int:
    return int(_next_candidate_id().split("-")[-1])


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


def _copy_base_train_to_candidate(
    base_train: Path,
    candidate_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    target = candidate_dir / "train.py"
    with BEST_TRAIN_LOCK:
        snapshot = _read_best_snapshot_unlocked()
        if not snapshot:
            snapshot = _write_best_snapshot_unlocked(
                best_candidate_id="baseline",
                best_val_bpb=None,
                source="copy_fallback",
            )
        shutil.copy2(base_train, target)
    return target, snapshot


def _plan_search_parent_id(plan: SearchPlan) -> str:
    return str(plan.search_parent_id or plan.parent_candidate_id or "").strip()


def _search_relation_type(operator: str) -> str:
    return {
        "exploit_best_mechanism": "exploit_after_same_direction",
        "pivot_near_miss": "pivot_from_same_direction",
        "avoid_failed_family": "avoid_failed_family_from_same_direction",
    }.get(operator, "search_parent")


def _normalized_candidate_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out[:8]


def _build_relation_edges(
    *,
    candidate_id: str,
    search_parent_id: str,
    search_operator: str,
    code_parent_id: str,
    nearest_candidate_ids: list[str] | None = None,
    duplicate_level: str = "",
) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(target: str, edge_type: str, source: str) -> None:
        target = str(target or "").strip()
        if not target or target == candidate_id:
            return
        key = (target, edge_type)
        if key in seen:
            return
        seen.add(key)
        edges.append({"target": target, "type": edge_type, "source": source})

    if search_parent_id:
        add(search_parent_id, _search_relation_type(search_operator), "search_plan")
    if code_parent_id and code_parent_id not in {"baseline", search_parent_id}:
        add(code_parent_id, "code_parent", "best_snapshot")
    for nearest_id in nearest_candidate_ids or []:
        if nearest_id == search_parent_id:
            continue
        add(
            nearest_id,
            duplicate_level or "nearest_candidate",
            "novelty_judge.nearest_candidate_ids",
        )
    return edges


def _lineage_updates(
    *,
    candidate_id: str,
    search_plan: SearchPlan,
    code_parent_snapshot: dict[str, Any],
    nearest_candidate_ids: list[str] | None = None,
    duplicate_level: str = "",
) -> dict[str, Any]:
    search_parent_id = _plan_search_parent_id(search_plan)
    code_parent_id = str(
        code_parent_snapshot.get("best_candidate_id") or "baseline"
    ).strip()
    code_parent_ref = str(
        code_parent_snapshot.get("best_train_path") or BEST_TRAIN_FILE.name
    )
    code_parent_hash = str(code_parent_snapshot.get("code_hash") or "")
    return {
        "search_parent_id": search_parent_id,
        "semantic_parent_id": search_parent_id,
        "code_parent_id": code_parent_id,
        "code_parent_ref": code_parent_ref,
        "code_parent_hash": code_parent_hash,
        "relation_edges": _build_relation_edges(
            candidate_id=candidate_id,
            search_parent_id=search_parent_id,
            search_operator=search_plan.operator,
            code_parent_id=code_parent_id,
            nearest_candidate_ids=nearest_candidate_ids,
            duplicate_level=duplicate_level,
        ),
    }


def _complete_search_plan_with_agent(
    *,
    agent: BaseCodeAgent,
    plan: SearchPlan,
    workdir: Path,
) -> SearchPlan:
    payload = plan_prompt_payload(plan)
    result = agent.complete_search_plan(
        plan_payload_json=payload,
        workdir=workdir,
    )
    _raise_if_fatal_model_error(result.stderr, result.content)
    (workdir / "llm_plan.log").write_text(result.content or "")
    if not result.success:
        _append_queue_event(
            "search_plan_completion_failed",
            {
                "plan_id": plan.plan_id,
                "reason": (result.stderr or "plan_agent_failed")[:200],
            },
        )
        return plan

    payload_obj = _parse_json_object(result.content)
    completed = complete_plan_from_payload(plan, payload_obj)
    ok, reason = validate_search_plan(completed)
    if not ok:
        _append_queue_event(
            "search_plan_validation_failed",
            {
                "plan_id": plan.plan_id,
                "reason": reason,
            },
        )
        return plan
    return completed


def _create_candidate(
    *,
    candidate_id: str,
    agent: BaseCodeAgent,
    base_train_path: Path,
    run_summaries: list[str],
    topic: str,
    search_plan: SearchPlan,
    max_duplicate_retries: int = 2,
) -> CandidateTask | None:
    candidate_dir = CANDIDATE_ROOT / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    forbidden_training_text = (
        "禁止运行任何训练命令或长任务命令，"
        "包括但不限于：uv run python train.py、python train.py、torchrun、nohup。"
        "训练只能由外层调度器执行。"
    )
    forbidden_git_history_text = (
        "禁止使用 git 命令查看、切换或回滚历史版本（如 git show/log/checkout/restore）。"
        "本轮历史实验信息只能来自提示词中的 generator_guidance。"
    )
    pivot_guidance = ""

    for attempt in range(1, max_duplicate_retries + 2):
        candidate_train, code_parent_snapshot = _copy_base_train_to_candidate(
            base_train_path,
            candidate_dir,
        )
        base_lineage = _lineage_updates(
            candidate_id=candidate_id,
            search_plan=search_plan,
            code_parent_snapshot=code_parent_snapshot,
        )
        base_code = candidate_train.read_text()
        generator_guidance = _refresh_generator_guidance(
            exclude_candidate_id=candidate_id,
        )
        search_plan_text = search_plan_to_prompt(search_plan)

        result = agent.refine(
            current_files={"train.py": base_code},
            run_summaries=run_summaries,
            metric_key="val_bpb",
            metric_direction="minimize",
            topic=topic,
            extra_hints=(
                "你不能自由选择研究方向。"
                "你必须执行 runner 给出的 SEARCH_PLAN。"
                "你只负责把该计划落实到 train.py。"
                "如果认为计划不可行，只能做最接近该计划的最小可运行实现，"
                "并在 DESCRIPTION 中说明偏差。\n"
                "## SEARCH_PLAN\n"
                f"{search_plan_text}\n"
                "禁止再读取 results.tsv、autorunner/artifacts、autorunner/candidates 或其它历史文件；"
                "以下 generator_guidance 是 runner 生成的权威建议上下文，"
                "并且已经包含 best、active、recent、failed 和 blocked_duplicate 方向。"
                "若方向近似，必须在 NOVELTY_CLAIM 和 DESCRIPTION 中说明关键差异。\n"
                f"{generator_guidance}\n"
                f"{pivot_guidance}"
                "选定一个方向后立即修改 train.py；py_compile 通过后立刻输出结构化字段，"
                "不要继续阅读、比较或复核其它文件。\n"
                "如果 SEARCH_PLAN.search_mode 是 mechanism_search，禁止只调数字常量、"
                "训练步数、学习率、batch size、dropout、hidden size、head 数或 depth；"
                "必须改变计算机制或计算路径。\n"
                f"{forbidden_training_text}\n"
                f"{forbidden_git_history_text}"
            ),
            workdir=candidate_dir,
        )
        _raise_if_fatal_model_error(result.stderr, result.content)

        (candidate_dir / f"llm_refine_attempt_{attempt}.log").write_text(
            result.content or ""
        )

        # 防止 agent 越权自行启动训练；若发现候选目录已有训练日志，直接废弃该候选。
        suspicious_logs = [
            candidate_dir / "run.log",
            candidate_dir / "run_repair_1.log",
            candidate_dir / "run_repair_2.log",
            candidate_dir / "run_repair_3.log",
        ]
        for p in suspicious_logs:
            if p.exists() and p.stat().st_size > 0:
                _update_registry_entry(
                    candidate_id,
                    status="generation_failed",
                    parent_ref=base_train_path.name,
                    **base_lineage,
                    description=f"candidate {candidate_id}",
                    failure_reason="agent_attempted_training",
                )
                _append_queue_event(
                    "candidate_generation_failed",
                    {
                        "candidate_id": candidate_id,
                        "reason": "agent_attempted_training",
                        "path": str(p.relative_to(WORKDIR)),
                    },
                )
                return None

        new_code = result.files.get("train.py", "").strip()
        field_source_content = result.content or ""
        salvaged_from_timeout = False

        if not result.success:
            failure_reason = (
                "generator_timeout" if result.timed_out else "generator_failed"
            )
            code_changed = bool(new_code and new_code != base_code.strip())
            if result.timed_out and code_changed:
                candidate_train.write_text(new_code)
                compile_ok, compile_error = _candidate_py_compile_ok(candidate_train)
                if compile_ok:
                    diff_summary = _code_diff_summary(base_code, new_code)
                    summary_result = agent.summarize_candidate(
                        candidate_id=candidate_id,
                        diff_summary=diff_summary,
                        workdir=candidate_dir,
                    )
                    _raise_if_fatal_model_error(
                        summary_result.stderr,
                        summary_result.content,
                    )
                    (candidate_dir / f"llm_summarize_attempt_{attempt}.log").write_text(
                        summary_result.content or ""
                    )
                    summary_fields = _extract_structured_refine_fields(
                        summary_result.content,
                        fallback=f"candidate {candidate_id}",
                    )
                    if summary_result.success and _structured_fields_are_usable(
                        summary_fields
                    ):
                        field_source_content = summary_result.content or ""
                        salvaged_from_timeout = True
                        _append_queue_event(
                            "candidate_generation_salvaged",
                            {
                                "candidate_id": candidate_id,
                                "attempt": attempt,
                                "reason": "timeout_with_compilable_diff",
                            },
                        )
                    else:
                        failure_reason = "timeout_salvage_summary_failed"
                else:
                    failure_reason = f"timeout_compile_failed: {compile_error}"

            if not salvaged_from_timeout:
                _update_registry_entry(
                    candidate_id,
                    status="generation_failed",
                    parent_ref=base_train_path.name,
                    **base_lineage,
                    description=f"candidate {candidate_id}",
                    failure_reason=(result.stderr or failure_reason)[:200],
                )
                _append_queue_event(
                    "candidate_generation_failed",
                    {
                        "candidate_id": candidate_id,
                        "reason": (result.stderr or failure_reason)[:200],
                    },
                )
                return None

        if not new_code:
            _update_registry_entry(
                candidate_id,
                status="generation_failed",
                parent_ref=base_train_path.name,
                **base_lineage,
                description=f"candidate {candidate_id}",
                failure_reason="train.py not updated",
            )
            _append_queue_event(
                "candidate_generation_failed",
                {
                    "candidate_id": candidate_id,
                    "reason": "train.py not updated",
                },
            )
            return None

        candidate_train.write_text(new_code)
        compile_ok, compile_error = _candidate_py_compile_ok(candidate_train)
        if not compile_ok:
            _update_registry_entry(
                candidate_id,
                status="generation_failed",
                parent_ref=base_train_path.name,
                **base_lineage,
                description=f"candidate {candidate_id}",
                failure_reason=f"py_compile_failed: {compile_error}"[:200],
            )
            _append_queue_event(
                "candidate_generation_failed",
                {
                    "candidate_id": candidate_id,
                    "reason": f"py_compile_failed: {compile_error}"[:200],
                },
            )
            return None

        fallback_desc = _extract_description_from_llm_reply(
            field_source_content,
            fallback=f"candidate {candidate_id}",
        )
        changed_keys = _changed_upper_keys(base_code, new_code)
        diff_summary = _code_diff_summary(base_code, new_code)
        structured = _extract_structured_refine_fields(
            field_source_content,
            fallback_desc,
        )
        if not _structured_fields_are_usable(structured):
            summary_result = agent.summarize_candidate(
                candidate_id=candidate_id,
                diff_summary=diff_summary,
                workdir=candidate_dir,
            )
            _raise_if_fatal_model_error(
                summary_result.stderr,
                summary_result.content,
            )
            (candidate_dir / f"llm_summarize_attempt_{attempt}.log").write_text(
                summary_result.content or ""
            )
            summary_structured = _extract_structured_refine_fields(
                summary_result.content,
                fallback=fallback_desc,
            )
            if summary_result.success and _structured_fields_are_usable(
                summary_structured
            ):
                structured = summary_structured
                field_source_content = summary_result.content or field_source_content
                _append_queue_event(
                    "candidate_generation_summarized",
                    {
                        "candidate_id": candidate_id,
                        "attempt": attempt,
                        "reason": "missing_structured_fields",
                    },
                )
            else:
                _update_registry_entry(
                    candidate_id,
                    status="generation_failed",
                    parent_ref=base_train_path.name,
                    **base_lineage,
                    description=f"candidate {candidate_id}",
                    failure_reason="missing_structured_fields",
                )
                _append_queue_event(
                    "candidate_generation_failed",
                    {
                        "candidate_id": candidate_id,
                        "reason": "missing_structured_fields",
                    },
                )
                return None

        desc = str(structured["description"])

        proposal = {
            "candidate_id": candidate_id,
            "status": "proposed",
            "parent_ref": base_train_path.name,
            "search_plan_id": search_plan.plan_id,
            "search_mode": search_plan.search_mode,
            "search_operator": search_plan.operator,
            **base_lineage,
            "target_direction_key": search_plan.target_direction_key,
            "intent": search_plan.intent,
            "rationale": search_plan.rationale,
            "avoid_direction_keys": search_plan.avoid_direction_keys,
            "search_plan": search_plan.to_dict(),
            "direction_key": structured["direction_key"],
            "hypothesis": structured["hypothesis"],
            "mechanism": structured["mechanism"],
            "touched_symbols": structured["touched_symbols"],
            "changed_upper_keys": changed_keys,
            "description": desc,
            "novelty_claim": structured["novelty_claim"],
            "generation_attempt": attempt,
            "novelty_judge": None,
            "failure_reason": None,
            "salvaged_from_timeout": salvaged_from_timeout,
            "reward": None,
            "reward_source": None,
        }
        proposal_updates = dict(proposal)
        proposal_updates.pop("candidate_id", None)
        _update_registry_entry(candidate_id, **proposal_updates)

        judge_context = _select_judge_context(
            candidate=proposal,
            registry_items=_load_experiment_registry(),
        )
        judge_result = agent.judge_novelty(
            candidate=proposal,
            registry_context=judge_context,
            diff_summary=diff_summary,
            workdir=candidate_dir,
        )
        _raise_if_fatal_model_error(judge_result.stderr, judge_result.content)
        (candidate_dir / f"llm_novelty_judge_attempt_{attempt}.log").write_text(
            judge_result.content or ""
        )

        judge_payload = _parse_json_object(judge_result.content)
        allow_run = False
        if judge_result.success and judge_payload is not None:
            allow_run = bool(judge_payload.get("allow_run"))
            if (
                search_plan.search_mode == "mechanism_search"
                and (
                    bool(judge_payload.get("is_pure_tuning"))
                    or judge_payload.get("mechanism_changed") is not True
                )
            ):
                allow_run = False
        duplicate_level = (
            str(judge_payload.get("duplicate_level") or "judge_failed")
            if judge_payload
            else "judge_failed"
        )
        nearest = (
            _normalized_candidate_ids(judge_payload.get("nearest_candidate_ids"))
            if isinstance(judge_payload, dict)
            else []
        )
        reason = (
            str(judge_payload.get("reason") or "")
            if isinstance(judge_payload, dict)
            else (judge_result.stderr or "novelty_judge_failed")
        )
        judge_lineage = _lineage_updates(
            candidate_id=candidate_id,
            search_plan=search_plan,
            code_parent_snapshot=code_parent_snapshot,
            nearest_candidate_ids=nearest,
            duplicate_level=duplicate_level,
        )

        if allow_run:
            task = CandidateTask(
                candidate_id=candidate_id,
                workdir=candidate_dir,
                train_py_path=candidate_train,
                refine_description=desc,
                parent_ref=base_train_path.name,
                created_at=time.time(),
                queued_at=time.time(),
                status="queued",
                direction_key=str(structured["direction_key"]),
                hypothesis=str(structured["hypothesis"]),
                mechanism=str(structured["mechanism"]),
                touched_symbols=list(structured["touched_symbols"]),
                novelty_claim=str(structured["novelty_claim"]),
                changed_upper_keys=changed_keys,
                search_plan_id=search_plan.plan_id,
                search_mode=search_plan.search_mode,
                search_operator=search_plan.operator,
                search_parent_id=str(judge_lineage.get("search_parent_id") or ""),
                semantic_parent_id=str(judge_lineage.get("semantic_parent_id") or ""),
                code_parent_id=str(judge_lineage.get("code_parent_id") or ""),
                code_parent_ref=str(judge_lineage.get("code_parent_ref") or ""),
                code_parent_hash=str(judge_lineage.get("code_parent_hash") or ""),
                relation_edges=list(judge_lineage.get("relation_edges") or []),
                target_direction_key=search_plan.target_direction_key,
                intent=search_plan.intent,
                rationale=search_plan.rationale,
                avoid_direction_keys=list(search_plan.avoid_direction_keys),
            )

            meta_payload = asdict(task)
            meta_payload["novelty_judge"] = judge_payload
            meta_payload["search_plan"] = search_plan.to_dict()
            (candidate_dir / "meta.json").write_text(
                json.dumps(meta_payload, indent=2, ensure_ascii=False, default=str)
            )

            _update_registry_entry(
                candidate_id,
                status="queued",
                novelty_judge=judge_payload,
                **judge_lineage,
                failure_reason=None,
                reward=None,
                reward_source=None,
            )
            _refresh_generator_guidance()
            _append_queue_event(
                "candidate_queued",
                {
                    "candidate_id": candidate_id,
                    "description": desc,
                    "direction_key": structured["direction_key"],
                    "search_plan_id": search_plan.plan_id,
                    "search_mode": search_plan.search_mode,
                    "search_operator": search_plan.operator,
                    "search_parent_id": judge_lineage.get("search_parent_id") or "",
                    "code_parent_id": judge_lineage.get("code_parent_id") or "",
                    "changed_upper_keys": changed_keys,
                    "novelty_level": duplicate_level,
                },
            )
            return task

        _update_registry_entry(
            candidate_id,
            status="blocked_duplicate",
            **judge_lineage,
            novelty_judge=judge_payload
            or {
                "allow_run": False,
                "duplicate_level": duplicate_level,
                "reason": reason,
                "innovation_type": search_plan.search_mode,
                "is_pure_tuning": None,
                "mechanism_changed": None,
            },
        )
        _refresh_generator_guidance()
        _append_queue_event(
            "candidate_blocked_duplicate",
            {
                "candidate_id": candidate_id,
                "attempt": attempt,
                "duplicate_level": duplicate_level,
                "nearest_candidate_ids": nearest,
                "reason": reason[:300],
            },
        )
        pivot = (
            str(judge_payload.get("required_pivot") or "")
            if isinstance(judge_payload, dict)
            else ""
        )
        if attempt > max_duplicate_retries:
            return None
        pivot_guidance = (
            "\n上一轮候选被 Novelty Judge 阻止，因为它与已有方向重复。"
            f"重复等级：{duplicate_level}。原因：{reason}。"
            f"你必须 pivot 到不同机制。建议：{pivot or '选择不同 direction_key 和 touched_symbols。'}\n"
        )

    return None


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
    on_process_started: Callable[[int], None] | None = None,
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
        on_process_started=on_process_started,
        env_overrides=_runtime_env_for_gpu(gpu_id),
    )
    return result


def _repair_candidate_if_needed(
    *,
    agent: BaseCodeAgent,
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
        _raise_if_fatal_model_error(repair_result.stderr, repair_result.content)

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
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    _mark_stale_active_registry_entries()
    _refresh_generator_guidance()

    agent = make_code_agent(
        backend=AGENT_BACKEND,
        model=MODEL if AGENT_BACKEND.strip().lower() == "claude" else None,
        timeout_sec=600,
        stream_output=False,
    )
    baseline_bpb = _load_baseline()
    _initialize_best_snapshot(baseline_bpb)

    queued: list[CandidateTask] = []
    running: dict[str, RunningTask] = {}
    generating_futures: set[Future[CandidateTask | None]] = set()
    completed_runs = 0
    executor = ThreadPoolExecutor(max_workers=max_running)
    generator_workers = max(1, min(queue_capacity, low_watermark))
    generator_executor = ThreadPoolExecutor(max_workers=generator_workers)
    failure_reason_counts: dict[str, int] = {}
    repair_stats = {"start": 0, "success": 0, "failed": 0, "out_of_scope": 0}
    last_no_gpu_event_ts = 0.0
    finished_recent: list[dict[str, str]] = []
    start_ts = time.time()
    state_lock = threading.Lock()
    candidate_id_lock = threading.Lock()
    next_candidate_num = _next_candidate_number()
    fatal_error_message: str | None = None
    fatal_alert_result: str = "not_sent"
    running_process_pids: dict[str, int] = {}

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

            def _trigger_fatal_abort(*, source: str) -> None:
                nonlocal fatal_error_message, fatal_alert_result
                if fatal_error_message:
                    return

                fatal_error_message = FATAL_STOP_REASON
                stop_refresh.set()

                with state_lock:
                    queued_ids = [x.candidate_id for x in queued]
                    queued.clear()
                    running_items = list(running.items())
                    pending_generation = list(generating_futures)

                for fut in pending_generation:
                    fut.cancel()

                aborted_ids = queued_ids
                for cid in aborted_ids:
                    _update_registry_entry(
                        cid,
                        status="aborted",
                        failure_reason=FATAL_STOP_REASON,
                        fatal_error=fatal_error_message,
                    )

                _append_queue_event(
                    "fatal_abort_triggered",
                    {
                        "source": source,
                        "aborted_candidates": aborted_ids[:50],
                        "running_untouched": [cid for cid, _ in running_items][:50],
                    },
                )

                alert_subject = "[AutoResearch] 实验已熔断停止：检测到模型API错误码"
                alert_body = (
                    f"time: {_now_iso()}\n"
                    f"model: {MODEL}\n"
                    f"backend: {AGENT_BACKEND}\n"
                    f"source: {source}\n"
                    f"aborted_queued_candidates: {len(aborted_ids)}\n"
                    f"running_left_untouched: {len(running_items)}\n"
                    f"workdir: {WORKDIR}\n"
                )
                sent, msg = _send_qq_alert_email(alert_subject, alert_body)
                fatal_alert_result = f"{'sent' if sent else 'failed'}: {msg}"[:300]
                _append_queue_event(
                    "fatal_alert_email",
                    {
                        "sent": sent,
                        "message": msg[:200],
                    },
                )

                fatal_payload = {
                    "ts": _now_iso(),
                    "reason": fatal_error_message,
                    "alert": fatal_alert_result,
                    "aborted_count": len(aborted_ids),
                }
                (ALERTS_DIR / "fatal_stop.json").write_text(
                    json.dumps(fatal_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

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
                        with state_lock:
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
                with state_lock:
                    return (
                        completed_runs,
                        baseline_bpb,
                        [],
                        {},
                        [],
                        dict(repair_stats),
                        dict(failure_reason_counts),
                    )

            def _generation_loop() -> None:
                nonlocal next_candidate_num
                interval = max(0.2, min(0.8, poll_interval_sec / 2))
                while not stop_refresh.is_set():
                    to_submit = 0
                    with state_lock:
                        queued_len = len(queued)
                        running_len = len(running)
                        generating_len = len(generating_futures)
                        active_or_queued = queued_len + running_len + generating_len
                        remaining = max_total_runs - (completed_runs + active_or_queued)
                        if (
                            remaining > 0
                            and queued_len + generating_len < low_watermark
                        ):
                            fill_target = low_watermark - (queued_len + generating_len)
                            capacity_left = queue_capacity - (
                                queued_len + generating_len
                            )
                            to_submit = max(
                                0, min(fill_target, capacity_left, remaining)
                            )

                    for _ in range(to_submit):
                        with candidate_id_lock:
                            candidate_id = f"cand-{next_candidate_num:06d}"
                            next_candidate_num += 1
                        registry_items = _load_experiment_registry()
                        search_plan = select_search_plan(
                            candidate_id=candidate_id,
                            registry_items=registry_items,
                            parent_ref=BEST_TRAIN_FILE.name,
                            parent_candidate_id=None,
                        )
                        search_parent_id = _plan_search_parent_id(search_plan)
                        search_relation_edges = _build_relation_edges(
                            candidate_id=candidate_id,
                            search_parent_id=search_parent_id,
                            search_operator=search_plan.operator,
                            code_parent_id="",
                        )
                        _update_registry_entry(
                            candidate_id,
                            status="proposed",
                            parent_ref=BEST_TRAIN_FILE.name,
                            description=f"{candidate_id} generation in progress",
                            direction_key="generation.in_progress",
                            search_plan_id=search_plan.plan_id,
                            search_mode=search_plan.search_mode,
                            search_operator=search_plan.operator,
                            search_parent_id=search_parent_id,
                            semantic_parent_id=search_parent_id,
                            target_direction_key=search_plan.target_direction_key,
                            intent=search_plan.intent,
                            rationale=search_plan.rationale,
                            avoid_direction_keys=search_plan.avoid_direction_keys,
                            relation_edges=search_relation_edges,
                            search_plan=search_plan.to_dict(),
                        )
                        candidate_dir = CANDIDATE_ROOT / candidate_id
                        candidate_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            search_plan = _complete_search_plan_with_agent(
                                agent=agent,
                                plan=search_plan,
                                workdir=candidate_dir,
                            )
                        except FatalModelAPIError:
                            _trigger_fatal_abort(source="plan_agent")
                            continue
                        except Exception as exc:
                            _append_queue_event(
                                "search_plan_completion_exception",
                                {
                                    "candidate_id": candidate_id,
                                    "error": f"{type(exc).__name__}: {exc}"[:300],
                                },
                            )
                        ok, reason = validate_search_plan(search_plan)
                        if not ok:
                            search_parent_id = _plan_search_parent_id(search_plan)
                            _update_registry_entry(
                                candidate_id,
                                status="generation_failed",
                                search_parent_id=search_parent_id,
                                semantic_parent_id=search_parent_id,
                                failure_reason=f"invalid_search_plan:{reason}"[:200],
                            )
                            _append_queue_event(
                                "candidate_generation_failed",
                                {
                                    "candidate_id": candidate_id,
                                    "reason": f"invalid_search_plan:{reason}"[:200],
                                },
                            )
                            continue
                        search_parent_id = _plan_search_parent_id(search_plan)
                        search_relation_edges = _build_relation_edges(
                            candidate_id=candidate_id,
                            search_parent_id=search_parent_id,
                            search_operator=search_plan.operator,
                            code_parent_id="",
                        )
                        _update_registry_entry(
                            candidate_id,
                            search_plan_id=search_plan.plan_id,
                            search_mode=search_plan.search_mode,
                            search_operator=search_plan.operator,
                            search_parent_id=search_parent_id,
                            semantic_parent_id=search_parent_id,
                            target_direction_key=search_plan.target_direction_key,
                            intent=search_plan.intent,
                            rationale=search_plan.rationale,
                            avoid_direction_keys=search_plan.avoid_direction_keys,
                            relation_edges=search_relation_edges,
                            search_plan=search_plan.to_dict(),
                            reward=None,
                            reward_source=None,
                        )
                        with state_lock:
                            run_summaries = _build_run_summaries()
                            fut = generator_executor.submit(
                                _create_candidate,
                                candidate_id=candidate_id,
                                agent=agent,
                                base_train_path=BEST_TRAIN_FILE,
                                run_summaries=run_summaries,
                                topic=topic,
                                search_plan=search_plan,
                            )
                            generating_futures.add(fut)

                    stop_refresh.wait(interval)

            def _dispatch_loop() -> None:
                nonlocal last_no_gpu_event_ts
                interval = max(0.2, min(0.8, poll_interval_sec / 2))
                while not stop_refresh.is_set():
                    queued_len = 0
                    with state_lock:
                        if fatal_error_message is not None:
                            can_dispatch = False
                            queued_len = len(queued)
                            running_len = len(running)
                            occupied = set()

                        else:
                            queued_len = len(queued)
                            running_len = len(running)
                            occupied = {
                                rt.task.gpu_id
                                for rt in running.values()
                                if rt.task.gpu_id is not None
                            }
                            can_dispatch = (
                                queued_len > 0
                                and running_len < max_running
                                and completed_runs + running_len < max_total_runs
                            )

                    if not can_dispatch:
                        stop_refresh.wait(interval)
                        continue

                    free_gpu_ids = _find_free_gpu_ids(
                        occupied_gpu_ids={int(x) for x in occupied},
                        mem_idle_threshold_mb=mem_idle_threshold_mb,
                        util_idle_threshold_pct=util_idle_threshold_pct,
                    )

                    if queued_len > 0 and not free_gpu_ids:
                        now_ts = time.time()
                        with state_lock:
                            if now_ts - last_no_gpu_event_ts >= 10.0 and queued:
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

                    if queued_len > 0 and not free_gpu_ids and dispatch_stall_sec > 0:
                        with state_lock:
                            oldest_wait = (
                                time.time() - min(x.queued_at for x in queued)
                                if queued
                                else 0.0
                            )
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

                    while free_gpu_ids:
                        with state_lock:
                            if (
                                not queued
                                or len(running) >= max_running
                                or completed_runs + len(running) >= max_total_runs
                            ):
                                break
                            gpu_id = free_gpu_ids.pop(0)
                            task = queued.pop(0)
                            task.status = "running"
                            task.gpu_id = gpu_id
                            task.launch_ts = time.time()

                            def _on_process_started(
                                pid: int, candidate_id: str = task.candidate_id
                            ) -> None:
                                with state_lock:
                                    running_process_pids[candidate_id] = pid

                            fut = executor.submit(
                                _run_one_candidate,
                                task=task,
                                gpu_id=gpu_id,
                                time_budget=time_budget,
                                on_process_started=_on_process_started,
                            )
                            running[task.candidate_id] = RunningTask(
                                task=task, future=fut
                            )
                            _update_registry_entry(
                                task.candidate_id,
                                status="running",
                                gpu_id=gpu_id,
                            )
                            _append_queue_event(
                                "candidate_running",
                                {
                                    "candidate_id": task.candidate_id,
                                    "gpu_id": gpu_id,
                                },
                            )

                    stop_refresh.wait(interval)

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
            generation_thread = threading.Thread(target=_generation_loop, daemon=True)
            generation_thread.start()
            dispatch_thread = threading.Thread(target=_dispatch_loop, daemon=True)
            dispatch_thread.start()

            while True:
                # 1) 回收生成完成的候选
                completed_gen: list[Future[CandidateTask | None]] = []
                with state_lock:
                    for fut in list(generating_futures):
                        if fut.done():
                            completed_gen.append(fut)
                            generating_futures.remove(fut)
                for fut in completed_gen:
                    try:
                        task = fut.result()
                    except FatalModelAPIError as exc:
                        _ = exc
                        _trigger_fatal_abort(source="generator")
                        task = None
                    except Exception as exc:
                        _append_queue_event(
                            "candidate_generation_exception",
                            {"error": f"{type(exc).__name__}: {exc}"[:300]},
                        )
                        task = None
                    if task is not None:
                        with state_lock:
                            queued.append(task)

                # 2) 回收完成任务
                with state_lock:
                    running_items = list(running.items())

                finished_ids: list[str] = []
                for cid, rt in running_items:
                    if not rt.future.done():
                        continue

                    try:
                        exp_result = rt.future.result()
                    except FatalModelAPIError as exc:
                        _ = exc
                        _trigger_fatal_abort(source="runner")
                        finished_ids.append(cid)
                        continue
                    except Exception as exc:
                        _append_queue_event(
                            "candidate_run_exception",
                            {
                                "candidate_id": cid,
                                "error": f"{type(exc).__name__}: {exc}"[:300],
                            },
                        )
                        finished_ids.append(cid)
                        continue

                    task = rt.task
                    finished_ids.append(cid)
                    with state_lock:
                        completed_runs += 1

                    task.status = "repairing"
                    if fatal_error_message is not None:
                        repaired = False
                        final_desc = task.refine_description
                        repair_status = "not_needed"
                        repair_attempts = 0
                    else:
                        try:
                            (
                                exp_result,
                                repaired,
                                final_desc,
                                repair_status,
                                repair_attempts,
                            ) = _repair_candidate_if_needed(
                                agent=agent,
                                task=task,
                                initial_result=exp_result,
                                gpu_id=task.gpu_id if task.gpu_id is not None else 0,
                                time_budget=time_budget,
                            )
                        except FatalModelAPIError as exc:
                            _ = exc
                            _trigger_fatal_abort(source="repair")
                            repaired = False
                            final_desc = task.refine_description
                            repair_status = "failed"
                            repair_attempts = 0
                        except Exception as exc:
                            repaired = False
                            final_desc = task.refine_description
                            repair_status = "failed"
                            repair_attempts = 0
                            _append_queue_event(
                                "candidate_repair_exception",
                                {
                                    "candidate_id": task.candidate_id,
                                    "error": f"{type(exc).__name__}: {exc}"[:300],
                                },
                            )
                    task.status = "running"
                    task.final_description = final_desc
                    with state_lock:
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
                        with state_lock:
                            baseline_bpb = exp_result.val_bpb
                        with BEST_TRAIN_LOCK:
                            shutil.copy2(task.train_py_path, BEST_TRAIN_FILE)
                            _write_best_snapshot_unlocked(
                                best_candidate_id=task.candidate_id,
                                best_val_bpb=exp_result.val_bpb,
                                source="candidate_keep",
                            )
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
                        with state_lock:
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

                    reward, reward_source = compute_reward(
                        baseline_before=baseline_before,
                        val_bpb=exp_result.val_bpb,
                        final_status=final_status,
                        run_status=exp_result.status,
                        discard_reason=discard_reason,
                    )
                    _update_registry_entry(
                        task.candidate_id,
                        status=final_status,
                        description=task.final_description or task.refine_description,
                        search_parent_id=task.search_parent_id,
                        semantic_parent_id=task.semantic_parent_id,
                        code_parent_id=task.code_parent_id,
                        code_parent_ref=task.code_parent_ref,
                        code_parent_hash=task.code_parent_hash,
                        relation_edges=task.relation_edges or [],
                        reward=round(reward, 6),
                        reward_source=reward_source,
                        parent_val_bpb=(
                            baseline_before
                            if math.isfinite(baseline_before)
                            else None
                        ),
                        result={
                            "run_status": exp_result.status,
                            "decision": decision,
                            "val_bpb": exp_result.val_bpb,
                            "peak_vram_mb": exp_result.peak_vram_mb,
                            "mfu_percent": exp_result.mfu_percent,
                            "repaired": repaired,
                            "repair_status": repair_status,
                            "discard_reason": discard_reason,
                        },
                    )
                    _refresh_generator_guidance()

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

                with state_lock:
                    for cid in finished_ids:
                        running.pop(cid, None)
                        running_process_pids.pop(cid, None)

                # 3) 终止条件
                with state_lock:
                    should_stop = (fatal_error_message is not None and not running) or (
                        completed_runs >= max_total_runs
                        and not running
                        and not queued
                        and not generating_futures
                    )
                if should_stop:
                    break

                time.sleep(poll_interval_sec)

            stop_refresh.set()
            generation_thread.join(timeout=1.0)
            dispatch_thread.join(timeout=1.0)
            refresh_thread.join(timeout=1.0)
    finally:
        generator_executor.shutdown(wait=True)
        executor.shutdown(wait=True)

    console.print(
        f"[runner] finished completed_runs={completed_runs} "
        f"best_val_bpb={baseline_bpb if baseline_bpb != float('inf') else 'inf'}"
    )
    if fatal_error_message is not None:
        console.print(
            "[runner] fatal abort triggered "
            f"reason={fatal_error_message} alert={fatal_alert_result}"
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
