"""Export AutoResearch search history as a graph artifact."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# 支持直接运行: python autorunner/search_tree_export.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from autorunner.env_config import load_dotenv, project_root


BASELINE_NODE_ID = "baseline"
CANDIDATE_ID_RE = re.compile(r"cand-\d+")
CANDIDATE_ID_FULL_RE = re.compile(r"^cand-(\d+)$")

SEARCH_MODE_LABELS_ZH = {
    "baseline_intake": "基线导入",
    "mechanism_search": "机制搜索",
    "control_tune": "控制调参",
    "local_tune_after_mechanism": "近失点局部微调",
    "pivot_after_failure": "失败后转向",
    "verify_winner": "优胜复验",
}

OPERATOR_LABELS_ZH = {
    "root": "根节点",
    "explore_new_mechanism": "探索新机制",
    "exploit_best_mechanism": "强化有效机制",
    "near_miss_refine": "近失点微调",
    "pivot_near_miss": "近似失败转向",
    "avoid_failed_family": "避开失败族",
    "control_tune": "控制调参",
    "verify_winner": "复验优胜分支",
    "merge_validate": "合并验证",
    "simplify_winner": "简化优胜分支",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
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
    return items


def _read_registry(path: Path) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for item in _read_jsonl(path):
        cid = str(item.get("candidate_id") or "").strip()
        if not cid:
            continue
        if cid not in merged:
            ordered_ids.append(cid)
            merged[cid] = {}
        merged[cid].update(item)
    return [merged[cid] for cid in ordered_ids]


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
    best_keep_bpb: float | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            cid = str(raw.get("commit") or raw.get("candidate_id") or "").strip()
            if not cid:
                continue

            val_bpb = _to_float(raw.get("val_bpb"))
            memory_gb = _to_float(raw.get("memory_gb"))
            status = str(raw.get("status") or "").strip()
            description = str(raw.get("description") or "").strip()

            reward: float | None = None
            reward_source = ""
            if val_bpb is not None and best_keep_bpb is not None and best_keep_bpb > 0:
                reward = (best_keep_bpb - val_bpb) / best_keep_bpb * 100.0
                reward_source = "results_relative_to_best_before_row"
            elif status == "keep":
                reward = 1.0
                reward_source = "results_first_keep_default"
            elif status == "discard":
                reward = -0.2
                reward_source = "results_discard_default"

            rows[cid] = {
                "candidate_id": cid,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "description": description,
                "reward": reward,
                "reward_source": reward_source,
            }

            if status == "keep" and val_bpb is not None:
                if best_keep_bpb is None or val_bpb < best_keep_bpb:
                    best_keep_bpb = val_bpb

    return rows


def _queue_status_from_event(event_type: str) -> str | None:
    return {
        "candidate_queued": "queued",
        "candidate_running": "running",
        "candidate_keep": "keep",
        "candidate_discard": "discard",
        "candidate_blocked_duplicate": "blocked_duplicate",
        "candidate_generation_failed": "generation_failed",
        "candidate_repair_exception": "repair_failed",
    }.get(event_type)


def _read_queue_events(path: Path) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for item in _read_jsonl(path):
        payload = item.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        cid = str(payload.get("candidate_id") or "").strip()
        if not cid:
            continue

        event_type = str(item.get("event") or "").strip()
        entry = by_candidate.setdefault(
            cid,
            {
                "candidate_id": cid,
                "event_counts": {},
                "events": [],
                "last_event": "",
                "last_event_ts": "",
                "last_event_payload": {},
            },
        )
        counts = entry["event_counts"]
        counts[event_type] = int(counts.get(event_type, 0)) + 1
        entry["last_event"] = event_type
        entry["last_event_ts"] = item.get("ts") or ""
        entry["last_event_payload"] = payload
        status = _queue_status_from_event(event_type)
        if status:
            entry["status"] = status
        entry["events"].append(
            {
                "ts": item.get("ts") or "",
                "event": event_type,
                "payload": payload,
            }
        )
    return by_candidate


def _result_dict(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result")
    return result if isinstance(result, dict) else {}


def _infer_parent_id(item: dict[str, Any]) -> str:
    cid = str(item.get("candidate_id") or "").strip()
    for key in (
        "search_parent_id",
        "parent_candidate_id",
        "parent_id",
        "source_candidate_id",
    ):
        value = str(item.get(key) or "").strip()
        if value and value != cid:
            return value

    parent_ref = str(item.get("parent_ref") or "").strip()
    match = CANDIDATE_ID_RE.search(parent_ref)
    if match:
        parent_id = match.group(0)
        if parent_id != cid:
            return parent_id

    return BASELINE_NODE_ID


def _status_default_reward(status: str) -> tuple[float | None, str]:
    if status == "keep":
        return 1.0, "status_keep_default"
    if status == "discard":
        return -0.2, "status_discard_default"
    if status == "blocked_duplicate":
        return -0.4, "status_blocked_duplicate_default"
    if status in {"generation_failed", "repair_failed"}:
        return -0.8, f"status_{status}_default"
    if status in {"queued", "running", "proposed"}:
        return None, "pending_no_reward"
    return None, "unknown_no_reward"


def _infer_reward(
    *,
    item: dict[str, Any],
    result_row: dict[str, Any] | None,
) -> tuple[float | None, str]:
    explicit = _to_float(item.get("reward"))
    if explicit is not None:
        return explicit, "registry_reward"

    result = _result_dict(item)
    parent_val = _to_float(item.get("parent_val_bpb"))
    val_bpb = _to_float(result.get("val_bpb"))
    if val_bpb is not None and parent_val is not None and parent_val > 0:
        return (parent_val - val_bpb) / parent_val * 100.0, "registry_parent_val_bpb"

    if result_row:
        reward = _to_float(result_row.get("reward"))
        if reward is not None:
            return reward, str(result_row.get("reward_source") or "results_reward")

    decision = str(result.get("decision") or "").strip()
    status = str(item.get("status") or (result_row or {}).get("status") or "").strip()
    if decision == "keep":
        status = "keep"
    elif decision == "discard" and not status:
        status = "discard"
    return _status_default_reward(status)


def _mix_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def _mix_color(start_hex: str, end_hex: str, t: float) -> str:
    start = start_hex.lstrip("#")
    end = end_hex.lstrip("#")
    sr, sg, sb = int(start[0:2], 16), int(start[2:4], 16), int(start[4:6], 16)
    er, eg, eb = int(end[0:2], 16), int(end[2:4], 16), int(end[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        _mix_channel(sr, er, t),
        _mix_channel(sg, eg, t),
        _mix_channel(sb, eb, t),
    )


def _reward_style(
    reward: float | None,
    *,
    max_abs_reward: float,
) -> dict[str, Any]:
    if reward is None:
        return {
            "reward_normalized": None,
            "reward_color": "#9ca3af",
            "reward_color_scale": "unknown_gray",
        }

    if reward == 0:
        return {
            "reward_normalized": 0.0,
            "reward_color": "#e5e7eb",
            "reward_color_scale": "zero_gray",
        }

    denom = max(max_abs_reward, 1e-9)
    intensity = min(1.0, abs(reward) / denom)
    if reward > 0:
        color = _mix_color("#dcfce7", "#166534", intensity)
        scale = "positive_green"
    else:
        color = _mix_color("#fee2e2", "#991b1b", intensity)
        scale = "negative_red"
    return {
        "reward_normalized": round(intensity if reward > 0 else -intensity, 6),
        "reward_color": color,
        "reward_color_scale": scale,
    }


def _status_color(status: str) -> str:
    return {
        "root": "#64748b",
        "keep": "#16a34a",
        "discard": "#dc2626",
        "running": "#2563eb",
        "queued": "#ca8a04",
        "proposed": "#a16207",
        "blocked_duplicate": "#9333ea",
        "generation_failed": "#6b7280",
        "repair_failed": "#7f1d1d",
    }.get(status, "#6b7280")


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _candidate_sort_key(candidate_id: Any) -> tuple[int, str]:
    text = str(candidate_id or "").strip()
    if text == BASELINE_NODE_ID:
        return (-1, text)
    match = CANDIDATE_ID_FULL_RE.match(text)
    if match:
        return (int(match.group(1)), text)
    return (10**9, text)


def _zh_label(mapping: dict[str, str], value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        return ""
    return mapping.get(key, key)


def _zh_display(raw_value: Any, label_value: Any) -> str:
    raw = str(raw_value or "").strip()
    label = str(label_value or "").strip()
    if not raw:
        return ""
    if not label or label == raw:
        return raw
    return f"{label} ({raw})"


def _build_raw_nodes(
    *,
    registry_items: list[dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {
        BASELINE_NODE_ID: {
            "id": BASELINE_NODE_ID,
            "parent_id": None,
            "label": "baseline",
            "status": "root",
            "reward": 0.0,
            "reward_source": "root",
            "val_bpb": None,
            "memory_gb": None,
            "direction_key": "baseline",
            "search_mode": "baseline_intake",
            "operator": "root",
            "description": "Initial baseline/root node.",
        }
    }

    all_ids = set(results_by_id) | set(events_by_id)
    for item in registry_items:
        cid = str(item.get("candidate_id") or "").strip()
        if cid:
            all_ids.add(cid)

    registry_by_id = {
        str(item.get("candidate_id") or "").strip(): item
        for item in registry_items
        if str(item.get("candidate_id") or "").strip()
    }

    for cid in sorted(all_ids):
        if cid == BASELINE_NODE_ID:
            continue
        item = dict(registry_by_id.get(cid) or {})
        result_row = results_by_id.get(cid)
        event_entry = events_by_id.get(cid)
        result = _result_dict(item)

        status = str(
            _coalesce(
                item.get("status"),
                result.get("decision"),
                (result_row or {}).get("status"),
                (event_entry or {}).get("status"),
                "unknown",
            )
        )
        val_bpb = _coalesce(
            result.get("val_bpb"),
            (result_row or {}).get("val_bpb"),
        )
        memory_gb = _coalesce(
            (result.get("peak_vram_mb") / 1024.0)
            if isinstance(result.get("peak_vram_mb"), (int, float))
            else None,
            (result_row or {}).get("memory_gb"),
        )
        reward, reward_source = _infer_reward(item=item, result_row=result_row)

        nodes[cid] = {
            "id": cid,
            "parent_id": _infer_parent_id(item),
            "label": cid,
            "status": status,
            "reward": round(reward, 6) if reward is not None else None,
            "reward_source": reward_source,
            "val_bpb": _to_float(val_bpb),
            "memory_gb": _to_float(memory_gb),
            "direction_key": _coalesce(
                item.get("target_direction_key"),
                item.get("direction_key"),
                (event_entry or {}).get("last_event_payload", {}).get("direction_key"),
                "unknown",
            ),
            "search_mode": _coalesce(item.get("search_mode"), item.get("stage"), ""),
            "operator": _coalesce(
                item.get("search_operator"),
                item.get("operator"),
                "",
            ),
            "search_plan_id": _coalesce(item.get("search_plan_id"), item.get("plan_id"), ""),
            "search_parent_id": _coalesce(
                item.get("search_parent_id"),
                item.get("parent_candidate_id"),
                "",
            ),
            "semantic_parent_id": item.get("semantic_parent_id") or "",
            "code_parent_id": item.get("code_parent_id") or "",
            "code_parent_ref": _coalesce(
                item.get("code_parent_ref"),
                item.get("parent_ref"),
                "",
            ),
            "lineage_relation_type": item.get("lineage_relation_type") or "",
            "lineage_inferred": bool(item.get("lineage_inferred")),
            "relation_edges": (
                item.get("relation_edges")
                if isinstance(item.get("relation_edges"), list)
                else []
            ),
            "parent_ref": item.get("parent_ref") or "",
            "epoch_id": item.get("epoch_id"),
            "search_depth": item.get("search_depth"),
            "intent": item.get("intent") or "",
            "rationale": item.get("rationale") or "",
            "hypothesis": item.get("hypothesis") or "",
            "mechanism": item.get("mechanism") or "",
            "description": _coalesce(
                item.get("description"),
                (result_row or {}).get("description"),
                (event_entry or {}).get("last_event_payload", {}).get("description"),
                "",
            ),
            "novelty_claim": item.get("novelty_claim") or "",
            "changed_upper_keys": item.get("changed_upper_keys") or [],
            "touched_symbols": item.get("touched_symbols") or [],
            "created_at": item.get("created_at") or "",
            "updated_at": item.get("updated_at") or "",
            "result": result,
            "novelty_judge": item.get("novelty_judge"),
            "failure_reason": _coalesce(
                item.get("failure_reason"),
                result.get("discard_reason"),
                result.get("failure_type"),
                "",
            ),
            "event_counts": (event_entry or {}).get("event_counts", {}),
            "last_event": (event_entry or {}).get("last_event", ""),
            "last_event_ts": (event_entry or {}).get("last_event_ts", ""),
        }

    for node in list(nodes.values()):
        parent_id = node.get("parent_id")
        if not parent_id or parent_id in nodes:
            continue
        nodes[str(parent_id)] = {
            "id": str(parent_id),
            "parent_id": BASELINE_NODE_ID,
            "label": str(parent_id),
            "status": "missing_parent",
            "reward": None,
            "reward_source": "missing_parent_placeholder",
            "val_bpb": None,
            "memory_gb": None,
            "direction_key": "unknown",
            "search_mode": "",
            "operator": "",
            "description": "Placeholder for missing parent node.",
        }

    return nodes


def _apply_display_labels(nodes_by_id: dict[str, dict[str, Any]]) -> None:
    for node in nodes_by_id.values():
        search_mode = node.get("search_mode")
        operator = node.get("operator")
        node["search_mode_label"] = _zh_label(SEARCH_MODE_LABELS_ZH, search_mode)
        node["operator_label"] = _zh_label(OPERATOR_LABELS_ZH, operator)


def _assign_depths(nodes: dict[str, dict[str, Any]]) -> None:
    def depth_for(node_id: str, visiting: set[str]) -> int:
        node = nodes.get(node_id)
        if not node:
            return 0
        existing = node.get("depth")
        if isinstance(existing, int):
            return existing
        if node_id == BASELINE_NODE_ID:
            node["depth"] = 0
            return 0
        parent_id = str(node.get("parent_id") or BASELINE_NODE_ID)
        if parent_id in visiting:
            node["depth"] = 1
            return 1
        visiting.add(node_id)
        parent_depth = depth_for(parent_id, visiting)
        visiting.remove(node_id)
        node["depth"] = parent_depth + 1
        return int(node["depth"])

    for node_id in list(nodes):
        depth_for(node_id, set())


def _finalize_tree_payload(
    *,
    workdir: Path,
    nodes_by_id: dict[str, dict[str, Any]],
    sources: dict[str, Any],
    demo: bool = False,
) -> dict[str, Any]:
    _assign_depths(nodes_by_id)
    _apply_display_labels(nodes_by_id)

    rewards = [
        float(node["reward"])
        for node in nodes_by_id.values()
        if _to_float(node.get("reward")) is not None and node["id"] != BASELINE_NODE_ID
    ]
    max_abs_reward = max((abs(x) for x in rewards), default=1.0)

    nodes: list[dict[str, Any]] = []
    for node in nodes_by_id.values():
        reward = _to_float(node.get("reward"))
        node.update(_reward_style(reward, max_abs_reward=max_abs_reward))
        node["status_color"] = _status_color(str(node.get("status") or ""))
        nodes.append(node)

    nodes.sort(key=lambda x: (int(x.get("depth") or 0), str(x.get("id") or "")))

    edges = []
    auxiliary_edges = []
    auxiliary_seen: set[tuple[str, str, str]] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        parent_id = node.get("parent_id")
        if not node_id or not parent_id:
            continue
        edges.append(
            {
                "id": f"{parent_id}->{node_id}",
                "source": parent_id,
                "target": node_id,
            }
        )
        for relation in node.get("relation_edges") or []:
            if not isinstance(relation, dict):
                continue
            target = str(relation.get("target") or "").strip()
            edge_type = str(relation.get("type") or "related").strip() or "related"
            if (
                not target
                or target == node_id
                or target == parent_id
                or target not in nodes_by_id
            ):
                continue
            key = (target, node_id, edge_type)
            if key in auxiliary_seen:
                continue
            auxiliary_seen.add(key)
            auxiliary_edges.append(
                {
                    "id": f"aux:{target}->{node_id}:{edge_type}",
                    "source": target,
                    "target": node_id,
                    "type": edge_type,
                    "source_field": str(relation.get("source") or ""),
                }
            )

    status_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    for node in nodes:
        status = str(node.get("status") or "unknown")
        direction = str(node.get("direction_key") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "workdir": str(workdir),
        "demo": demo,
        "sources": sources,
        "reward": {
            "meaning": "Positive is better; negative is worse. Explicit registry reward is used when available, otherwise inferred from parent_val_bpb, results.tsv, or status defaults.",
            "max_abs_reward": round(max_abs_reward, 6),
            "color_scale": {
                "positive": "light-to-dark green",
                "negative": "light-to-dark red",
                "unknown": "gray",
            },
        },
        "display_labels": {
            "search_mode": SEARCH_MODE_LABELS_ZH,
            "operator": OPERATOR_LABELS_ZH,
        },
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "auxiliary_edge_count": len(auxiliary_edges),
            "status_counts": status_counts,
            "direction_counts": direction_counts,
        },
        "nodes": nodes,
        "edges": edges,
        "auxiliary_edges": auxiliary_edges,
    }


def build_search_tree(
    *,
    workdir: Path,
) -> dict[str, Any]:
    artifacts_dir = workdir / "autorunner" / "artifacts"
    state_dir = artifacts_dir / "state"
    registry_file = _first_existing_path(
        state_dir / "experiment_registry.jsonl",
        artifacts_dir / "experiment_registry.jsonl",
    )
    queue_events_file = _first_existing_path(
        state_dir / "parallel_queue.jsonl",
        artifacts_dir / "parallel_queue.jsonl",
    )
    results_file = workdir / "results.tsv"

    registry_items = _read_registry(registry_file)
    results_by_id = _read_results(results_file)
    events_by_id = _read_queue_events(queue_events_file)
    nodes_by_id = _build_raw_nodes(
        registry_items=registry_items,
        results_by_id=results_by_id,
        events_by_id=events_by_id,
    )
    return _finalize_tree_payload(
        workdir=workdir,
        nodes_by_id=nodes_by_id,
        sources={
            "registry_file": str(registry_file),
            "registry_items": len(registry_items),
            "results_file": str(results_file),
            "results_items": len(results_by_id),
            "queue_events_file": str(queue_events_file),
            "queue_event_candidates": len(events_by_id),
        },
    )


def build_demo_search_tree(*, workdir: Path) -> dict[str, Any]:
    nodes_by_id: dict[str, dict[str, Any]] = {
        BASELINE_NODE_ID: {
            "id": BASELINE_NODE_ID,
            "parent_id": None,
            "label": "baseline",
            "status": "root",
            "reward": 0.0,
            "reward_source": "demo_root",
            "val_bpb": 1.235,
            "direction_key": "baseline",
            "search_mode": "baseline_intake",
            "operator": "root",
            "description": "Demo baseline node. User-provided train.py starts here.",
        },
        "cand-000001": {
            "id": "cand-000001",
            "parent_id": BASELINE_NODE_ID,
            "label": "cand-000001",
            "status": "keep",
            "reward": 2.4,
            "reward_source": "demo",
            "val_bpb": 1.205,
            "direction_key": "architecture.attention",
            "search_mode": "mechanism_search",
            "operator": "explore_new_mechanism",
            "intent": "Try a mechanism-level attention path change.",
            "rationale": "Attention has low visit count and broad upside.",
            "description": "Add a lightweight attention mixing mechanism while keeping width/depth fixed.",
        },
        "cand-000002": {
            "id": "cand-000002",
            "parent_id": BASELINE_NODE_ID,
            "label": "cand-000002",
            "status": "discard",
            "reward": -0.7,
            "reward_source": "demo",
            "val_bpb": 1.244,
            "direction_key": "optimizer.lr_schedule",
            "search_mode": "control_tune",
            "operator": "control_tune",
            "intent": "Stabilize baseline with a small LR schedule adjustment.",
            "rationale": "Useful as control, but not expected to be a mechanism-level innovation.",
            "description": "Tune warmup and LR decay; result did not improve.",
        },
        "cand-000003": {
            "id": "cand-000003",
            "parent_id": BASELINE_NODE_ID,
            "label": "cand-000003",
            "status": "blocked_duplicate",
            "reward": -1.0,
            "reward_source": "demo",
            "val_bpb": None,
            "direction_key": "architecture.width_depth",
            "search_mode": "mechanism_search",
            "operator": "avoid_failed_family",
            "description": "Blocked because it only repeated width/depth scaling.",
            "novelty_judge": {
                "allow_run": False,
                "is_pure_tuning": True,
                "mechanism_changed": False,
                "reason": "Pure capacity scaling is not allowed in mechanism_search.",
            },
        },
        "cand-000004": {
            "id": "cand-000004",
            "parent_id": BASELINE_NODE_ID,
            "label": "cand-000004",
            "status": "running",
            "reward": None,
            "reward_source": "demo_pending",
            "val_bpb": None,
            "direction_key": "regularization.norm",
            "search_mode": "mechanism_search",
            "operator": "explore_new_mechanism",
            "description": "Currently running demo node with unknown reward.",
        },
        "cand-000005": {
            "id": "cand-000005",
            "parent_id": "cand-000001",
            "label": "cand-000005",
            "status": "keep",
            "reward": 4.8,
            "reward_source": "demo",
            "val_bpb": 1.177,
            "direction_key": "architecture.attention",
            "search_mode": "pivot_after_failure",
            "operator": "pivot_near_miss",
            "intent": "Keep the attention family but replace the concrete mechanism.",
            "rationale": "The first attention node improved; this child tests a sharper mechanism.",
            "description": "Pivot from generic mixing to gated local attention residual.",
        },
        "cand-000006": {
            "id": "cand-000006",
            "parent_id": "cand-000001",
            "label": "cand-000006",
            "status": "generation_failed",
            "reward": -2.0,
            "reward_source": "demo",
            "val_bpb": None,
            "direction_key": "data.curriculum",
            "search_mode": "mechanism_search",
            "operator": "explore_new_mechanism",
            "failure_reason": "py_compile_failed",
            "description": "Demo failed child: generated code did not compile.",
        },
        "cand-000007": {
            "id": "cand-000007",
            "parent_id": "cand-000005",
            "label": "cand-000007",
            "status": "queued",
            "reward": None,
            "reward_source": "demo_pending",
            "val_bpb": None,
            "direction_key": "architecture.attention",
            "search_mode": "verify_winner",
            "operator": "verify_winner",
            "description": "Queued verification run for the best attention candidate.",
        },
    }
    return _finalize_tree_payload(
        workdir=workdir,
        nodes_by_id=nodes_by_id,
        sources={
            "demo": True,
            "registry_file": "",
            "registry_items": 0,
            "results_file": "",
            "results_items": 0,
            "queue_events_file": "",
            "queue_event_candidates": 0,
        },
        demo=True,
    )


def _layout_tree(tree: dict[str, Any]) -> tuple[dict[str, tuple[int, int]], int, int]:
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    children: dict[str, list[str]] = {}
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_by_id[node_id] = node
        parent_id = node.get("parent_id")
        if parent_id:
            children.setdefault(str(parent_id), []).append(node_id)

    for ids in children.values():
        ids.sort()

    x_gap = 260
    y_gap = 150
    margin_x = 100
    margin_y = 80
    positions: dict[str, tuple[int, int]] = {}
    next_leaf = 0

    def place(node_id: str, depth: int, visiting: set[str]) -> float:
        nonlocal next_leaf
        if node_id in visiting:
            y = next_leaf * y_gap
            next_leaf += 1
            return y
        visiting.add(node_id)
        child_ids = [cid for cid in children.get(node_id, []) if cid in node_by_id]
        if not child_ids:
            y = next_leaf * y_gap
            next_leaf += 1
        else:
            ys = [place(cid, depth + 1, visiting) for cid in child_ids]
            y = sum(ys) / len(ys)
        visiting.remove(node_id)
        positions[node_id] = (margin_x + depth * x_gap, int(margin_y + y))
        return y

    roots = [
        str(node.get("id"))
        for node in nodes
        if isinstance(node, dict) and not node.get("parent_id")
    ] or [BASELINE_NODE_ID]
    for root_id in roots:
        if root_id in node_by_id:
            place(root_id, 0, set())

    for node_id, node in node_by_id.items():
        if node_id in positions:
            continue
        depth = int(node.get("depth") or 0)
        positions[node_id] = (margin_x + depth * x_gap, margin_y + next_leaf * y_gap)
        next_leaf += 1

    max_x = max((x for x, _ in positions.values()), default=margin_x) + margin_x + 180
    max_y = max((y for _, y in positions.values()), default=margin_y) + margin_y + 90
    return positions, max_x, max_y


def _node_tooltip(node: dict[str, Any]) -> str:
    mode_display = _zh_display(
        node.get("search_mode"),
        node.get("search_mode_label"),
    )
    operator_display = _zh_display(
        node.get("operator"),
        node.get("operator_label"),
    )
    search_parent_display = node.get("parent_id") or ""
    code_parent_display = (
        _coalesce(node.get("code_parent_id"), node.get("code_parent_ref"), "") or ""
    )
    parts = [
        f"id: {node.get('id')}",
        f"search_parent: {search_parent_display}",
        f"code_parent: {code_parent_display}",
        f"status: {node.get('status')}",
        f"reward: {node.get('reward')}",
        f"val_bpb: {node.get('val_bpb')}",
        f"direction: {node.get('direction_key')}",
        f"mode: {mode_display}",
        f"operator: {operator_display}",
        f"description: {node.get('description')}",
    ]
    return "\n".join(str(x) for x in parts if x is not None)


def _table_group_root_id(
    node: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> str:
    node_id = str(node.get("id") or "")
    if not node_id or node_id == BASELINE_NODE_ID:
        return BASELINE_NODE_ID

    current = node
    root_id = node_id
    visited: set[str] = set()
    while True:
        current_id = str(current.get("id") or "")
        if current_id in visited:
            return root_id
        visited.add(current_id)

        parent_id = str(current.get("parent_id") or "")
        if not parent_id or parent_id == BASELINE_NODE_ID:
            return root_id
        parent = node_by_id.get(parent_id)
        if not parent:
            return root_id
        root_id = parent_id
        current = parent


def _table_group_label(
    root_id: str,
    rows: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> str:
    if root_id == BASELINE_NODE_ID:
        return "基线"
    root_node = node_by_id.get(root_id) or {}
    direction = str(
        _coalesce(
            root_node.get("direction_key"),
            rows[0].get("direction_key") if rows else "",
            "unknown",
        )
    )
    return f"相关组：{direction} · root {root_id} · {len(rows)} nodes"


def render_search_tree_html(tree: dict[str, Any]) -> str:
    positions, width, height = _layout_tree(tree)
    nodes = [node for node in tree.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in tree.get("edges", []) if isinstance(edge, dict)]
    node_by_id = {str(node.get("id") or ""): node for node in nodes}
    summary = tree.get("summary") if isinstance(tree.get("summary"), dict) else {}

    edge_svg: list[str] = []
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        edge_svg.append(
            f'<path class="main-edge" d="M {sx + 62} {sy} C {sx + 150} {sy}, {tx - 150} {ty}, {tx - 62} {ty}" />'
        )

    node_svg: list[str] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        fill = str(node.get("reward_color") or "#9ca3af")
        stroke = str(node.get("status_color") or "#6b7280")
        status = str(node.get("status") or "")
        reward = node.get("reward")
        val_bpb = node.get("val_bpb")
        direction = str(node.get("direction_key") or "")
        label = html.escape(str(node.get("label") or node_id))
        subtitle = html.escape(direction[:28])
        reward_text = "reward ?" if reward is None else f"reward {float(reward):+.2f}"
        metric_text = "" if val_bpb is None else f"val {float(val_bpb):.4f}"
        tooltip = html.escape(_node_tooltip(node))
        node_svg.append(
            "\n".join(
                [
                    f'<g class="node" transform="translate({x},{y})">',
                    f"<title>{tooltip}</title>",
                    f'<rect x="-72" y="-42" width="144" height="84" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
                    f'<text class="node-title" x="0" y="-17">{label}</text>',
                    f'<text class="node-subtitle" x="0" y="4">{html.escape(status)}</text>',
                    f'<text class="node-small" x="0" y="22">{html.escape(reward_text)}</text>',
                    f'<text class="node-small" x="0" y="38">{html.escape(metric_text or subtitle)}</text>',
                    "</g>",
                ]
            )
        )

    grouped_nodes: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        root_id = _table_group_root_id(node, node_by_id)
        grouped_nodes.setdefault(root_id, []).append(node)

    details_rows: list[str] = []
    table_column_count = 10
    for root_id in sorted(grouped_nodes, key=_candidate_sort_key):
        group_rows = sorted(
            grouped_nodes[root_id],
            key=lambda item: _candidate_sort_key(item.get("id")),
        )
        group_label = _table_group_label(root_id, group_rows, node_by_id)
        details_rows.append(
            '<tr class="group-row">'
            f'<td colspan="{table_column_count}">{html.escape(group_label)}</td>'
            "</tr>"
        )
        for node in group_rows:
            mode_display = _zh_display(
                node.get("search_mode"),
                node.get("search_mode_label"),
            )
            operator_display = _zh_display(
                node.get("operator"),
                node.get("operator_label"),
            )
            code_parent_display = _coalesce(
                node.get("code_parent_id"),
                node.get("code_parent_ref"),
                "",
            )
            details_rows.append(
                "<tr>"
                f"<td>{html.escape(str(node.get('id') or ''))}</td>"
                f"<td>{html.escape(str(node.get('parent_id') or ''))}</td>"
                f"<td>{html.escape(str(code_parent_display or ''))}</td>"
                f"<td>{html.escape(str(node.get('status') or ''))}</td>"
                f"<td>{html.escape(str(node.get('reward') if node.get('reward') is not None else ''))}</td>"
                f"<td>{html.escape(str(node.get('val_bpb') if node.get('val_bpb') is not None else ''))}</td>"
                f"<td>{html.escape(str(node.get('direction_key') or ''))}</td>"
                f"<td>{html.escape(mode_display)}</td>"
                f"<td>{html.escape(operator_display)}</td>"
                f"<td>{html.escape(str(node.get('description') or ''))}</td>"
                "</tr>"
            )

    demo_note = (
        '<span class="badge">DEMO DATA</span>'
        if bool(tree.get("demo"))
        else '<span class="badge live">REAL DATA</span>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoResearch Search Tree</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    body {{
      margin: 0;
      background: #f8fafc;
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid #d9e2ec;
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: #475569;
      font-size: 13px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border: 1px solid #c084fc;
      border-radius: 6px;
      color: #6b21a8;
      background: #faf5ff;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.live {{
      border-color: #86efac;
      color: #166534;
      background: #f0fdf4;
    }}
	    main {{
	      padding: 18px 24px 32px;
	    }}
	    .tabs {{
	      display: flex;
	      flex-wrap: wrap;
	      gap: 6px;
	      margin-bottom: 14px;
	      border-bottom: 1px solid #d9e2ec;
	    }}
	    .tab-button {{
	      appearance: none;
	      border: 1px solid transparent;
	      border-bottom: 0;
	      background: transparent;
	      color: #475569;
	      padding: 9px 14px;
	      font: inherit;
	      font-size: 13px;
	      font-weight: 700;
	      cursor: pointer;
	    }}
	    .tab-button:hover {{
	      color: #0f172a;
	      background: #f8fafc;
	    }}
	    .tab-button[aria-selected="true"] {{
	      color: #0f172a;
	      background: #ffffff;
	      border-color: #d9e2ec;
	    }}
	    .tab-panel {{
	      display: none;
	    }}
	    .tab-panel.active {{
	      display: block;
	    }}
	    .legend {{
	      display: flex;
	      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
      color: #475569;
      font-size: 13px;
    }}
	    .swatch {{
	      display: inline-block;
	      width: 16px;
      height: 16px;
      border-radius: 4px;
      border: 1px solid #94a3b8;
	      vertical-align: text-bottom;
	      margin-right: 4px;
	    }}
	    .line-sample {{
	      display: inline-block;
	      width: 28px;
	      height: 0;
	      border-top: 2px solid #94a3b8;
	      vertical-align: middle;
	      margin-right: 4px;
	    }}
	    .canvas {{
      overflow: auto;
      border: 1px solid #d9e2ec;
      background: #ffffff;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    svg {{
      display: block;
      min-width: 100%;
    }}
	    path.main-edge {{
	      fill: none;
	      stroke: #94a3b8;
	      stroke-width: 2;
	    }}
    .node-title {{
      font-size: 14px;
      font-weight: 700;
      text-anchor: middle;
      fill: #0f172a;
    }}
    .node-subtitle {{
      font-size: 12px;
      font-weight: 600;
      text-anchor: middle;
      fill: #1f2937;
    }}
    .node-small {{
      font-size: 11px;
      text-anchor: middle;
      fill: #334155;
    }}
    section {{
      margin-top: 18px;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      vertical-align: top;
    }}
	    th {{
	      background: #f1f5f9;
	      font-weight: 700;
	      color: #334155;
	    }}
	    .group-row td {{
	      background: #e2e8f0;
	      color: #0f172a;
	      font-weight: 800;
	      border-top: 2px solid #cbd5e1;
	    }}
	    td {{
	      color: #334155;
	    }}
  </style>
</head>
<body>
  <header>
    <h1>AutoResearch Search Tree</h1>
    <div class="meta">
      {demo_note}
	      <span>generated_at: {html.escape(str(tree.get("generated_at") or ""))}</span>
	      <span>nodes: {html.escape(str(summary.get("node_count") or 0))}</span>
	      <span>edges: {html.escape(str(summary.get("edge_count") or 0))}</span>
	    </div>
	  </header>
	  <main>
	    <div class="tabs" role="tablist" aria-label="Search tree views">
	      <button class="tab-button" id="tab-tree" type="button" role="tab" aria-controls="panel-tree" aria-selected="true" data-tab-target="panel-tree">搜索树</button>
	      <button class="tab-button" id="tab-table" type="button" role="tab" aria-controls="panel-table" aria-selected="false" data-tab-target="panel-table">表格信息</button>
	    </div>
	    <section id="panel-tree" class="tab-panel active" role="tabpanel" aria-labelledby="tab-tree">
	      <div class="legend">
	        <span><span class="swatch" style="background:#166534"></span>positive reward</span>
	        <span><span class="swatch" style="background:#991b1b"></span>negative reward</span>
	        <span><span class="swatch" style="background:#9ca3af"></span>unknown reward</span>
	        <span><span class="line-sample"></span>search parent</span>
	        <span>border color indicates status</span>
	      </div>
	      <div class="canvas">
	        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="AutoResearch search tree">
	          <g class="edges">
	            {"".join(edge_svg)}
	          </g>
	          <g class="nodes">
	            {"".join(node_svg)}
	          </g>
	        </svg>
	      </div>
	    </section>
	    <section id="panel-table" class="tab-panel" role="tabpanel" aria-labelledby="tab-table">
	      <h2>节点详情</h2>
	      <table>
        <thead>
            <tr>
              <th>ID</th>
            <th>搜索父节点</th>
            <th>代码父节点</th>
            <th>状态</th>
            <th>Reward</th>
            <th>Val BPB</th>
            <th>方向</th>
            <th>模式</th>
            <th>算子</th>
            <th>描述</th>
          </tr>
        </thead>
        <tbody>
          {"".join(details_rows)}
        </tbody>
	      </table>
	    </section>
	  </main>
	  <script>
	    const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
	    const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
	    for (const button of tabButtons) {{
	      button.addEventListener("click", () => {{
	        const targetId = button.getAttribute("data-tab-target");
	        for (const item of tabButtons) {{
	          item.setAttribute("aria-selected", String(item === button));
	        }}
	        for (const panel of tabPanels) {{
	          panel.classList.toggle("active", panel.id === targetId);
	        }}
	      }});
	    }}
	  </script>
	</body>
	</html>
	"""


def export_search_tree_html(
    *,
    tree: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(render_search_tree_html(tree), encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path


def export_search_tree(
    *,
    workdir: Path | None = None,
    output_path: Path | None = None,
    html_output_path: Path | None = None,
    demo: bool = False,
) -> Path:
    root = workdir or Path(os.getenv("AR_WORKDIR", str(project_root()))).expanduser()
    root = root.resolve()
    tree = build_demo_search_tree(workdir=root) if demo else build_search_tree(workdir=root)
    tree_dir = root / "autorunner" / "artifacts" / "tree"
    out = output_path or (tree_dir / ("search_tree_demo.json" if demo else "search_tree.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(tree, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(out)
    if html_output_path is not None:
        export_search_tree_html(tree=tree, output_path=html_output_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export AutoResearch registry/results as search_tree.json."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Workspace root. Defaults to AR_WORKDIR or project root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to autorunner/artifacts/tree/search_tree.json.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional output HTML path for a static visual graph.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Export a synthetic demo tree instead of reading history files.",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    output_path = export_search_tree(
        workdir=args.workdir,
        output_path=args.output,
        html_output_path=args.html_output,
        demo=args.demo,
    )
    print(output_path)
    if args.html_output is not None:
        print(args.html_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
