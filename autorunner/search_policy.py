"""Search policy primitives for AutoResearch candidate planning."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_OPERATORS = (
    "explore_new_mechanism",
    "exploit_best_mechanism",
    "pivot_near_miss",
    "avoid_failed_family",
)
DEFAULT_DIRECTION_POOL = (
    "architecture.attention",
    "architecture.mlp",
    "architecture.residual_path",
    "regularization.norm",
    "regularization.objective",
    "optimizer.update_rule",
    "data.curriculum",
    "training.stability",
)
PURE_TUNING_KEYWORDS = {
    "lr",
    "learning_rate",
    "batch",
    "dropout",
    "warmup",
    "weight_decay",
    "hidden_size",
    "n_embd",
    "n_layer",
    "n_head",
    "depth",
    "width",
    "aspect_ratio",
}


@dataclass
class SearchPlan:
    plan_id: str
    parent_candidate_id: str | None
    parent_ref: str
    operator: str
    target_direction_key: str
    intent: str
    rationale: str
    avoid_direction_keys: list[str]
    priority: float
    search_mode: str = "mechanism_search"
    experiment_brief: str = ""
    expected_mechanism: str = ""
    success_interpretation: str = ""
    failure_interpretation: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("priority", None)
        return data


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


def _status(item: dict[str, Any]) -> str:
    return str(item.get("status") or "").strip()


def _direction(item: dict[str, Any]) -> str:
    return str(
        item.get("target_direction_key")
        or item.get("direction_key")
        or "unknown"
    ).strip()


def _is_keep(item: dict[str, Any]) -> bool:
    result = _result_dict(item)
    return _status(item) == "keep" or str(result.get("decision") or "") == "keep"


def _is_discard(item: dict[str, Any]) -> bool:
    result = _result_dict(item)
    return _status(item) == "discard" or str(result.get("decision") or "") == "discard"


def summarize_registry(registry_items: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction: dict[str, dict[str, Any]] = {}
    active_statuses = {"proposed", "queued", "running"}

    for direction in DEFAULT_DIRECTION_POOL:
        by_direction[direction] = {
            "direction_key": direction,
            "visits": 0,
            "keep_count": 0,
            "discard_count": 0,
            "blocked_count": 0,
            "active_count": 0,
            "best_val_bpb": None,
            "recent_failure_reasons": [],
            "last_candidate_id": "",
        }

    for item in registry_items:
        direction = _direction(item)
        stats = by_direction.setdefault(
            direction,
            {
                "direction_key": direction,
                "visits": 0,
                "keep_count": 0,
                "discard_count": 0,
                "blocked_count": 0,
                "active_count": 0,
                "best_val_bpb": None,
                "recent_failure_reasons": [],
                "last_candidate_id": "",
            },
        )
        stats["visits"] += 1
        stats["last_candidate_id"] = str(item.get("candidate_id") or "")
        status = _status(item)
        if status in active_statuses:
            stats["active_count"] += 1
        if _is_keep(item):
            stats["keep_count"] += 1
        if _is_discard(item):
            stats["discard_count"] += 1
        if status == "blocked_duplicate":
            stats["blocked_count"] += 1

        result = _result_dict(item)
        reason = str(
            item.get("failure_reason")
            or result.get("discard_reason")
            or result.get("failure_type")
            or ""
        ).strip()
        if reason:
            stats["recent_failure_reasons"].append(reason)
            stats["recent_failure_reasons"] = stats["recent_failure_reasons"][-5:]

        val_bpb = _result_val_bpb(item)
        if val_bpb is not None:
            best = stats["best_val_bpb"]
            if best is None or val_bpb < best:
                stats["best_val_bpb"] = val_bpb

    active_directions = [
        k for k, v in by_direction.items() if int(v.get("active_count") or 0) > 0
    ]
    blocked_directions = [
        k for k, v in by_direction.items() if int(v.get("blocked_count") or 0) > 0
    ]
    return {
        "by_direction": by_direction,
        "active_directions": sorted(active_directions),
        "blocked_directions": sorted(blocked_directions),
        "total_items": len(registry_items),
    }


def _operator_for_direction(stats: dict[str, Any]) -> str:
    visits = int(stats.get("visits") or 0)
    keep_count = int(stats.get("keep_count") or 0)
    discard_count = int(stats.get("discard_count") or 0)
    blocked_count = int(stats.get("blocked_count") or 0)
    if blocked_count >= 2 or discard_count >= 3:
        return "avoid_failed_family"
    if visits == 0:
        return "explore_new_mechanism"
    if keep_count > 0:
        return "exploit_best_mechanism"
    return "pivot_near_miss"


def _choose_direction(summary: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    by_direction = summary.get("by_direction")
    if not isinstance(by_direction, dict) or not by_direction:
        return "architecture.attention", {}, 1.0

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for direction, stats in by_direction.items():
        visits = int(stats.get("visits") or 0)
        keep_count = int(stats.get("keep_count") or 0)
        active_count = int(stats.get("active_count") or 0)
        blocked_count = int(stats.get("blocked_count") or 0)
        discard_count = int(stats.get("discard_count") or 0)
        keep_rate = keep_count / max(1, visits)
        exploration = 1.0 / math.sqrt(max(1, visits))
        score = keep_rate * 2.0 + exploration
        score -= active_count * 2.0
        score -= blocked_count * 0.8
        score -= discard_count * 0.25
        scored.append((score, str(direction), stats))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0]
    if random.random() < 0.15 and len(scored) > 1:
        top = random.choice(scored[: min(4, len(scored))])
    return top[1], top[2], float(top[0])


def _default_intent(search_mode: str, operator: str, direction: str) -> str:
    if search_mode == "control_tune":
        return (
            f"Tune only control hyperparameters for {direction} to stabilize "
            "training and keep the parent node comparable."
        )
    if operator == "pivot_near_miss":
        return (
            f"Keep the broad {direction} goal, but replace the concrete mechanism "
            "with a mechanism-level alternative."
        )
    if operator == "exploit_best_mechanism":
        return (
            f"Make a focused mechanism-level refinement to the best known "
            f"{direction} branch without pure capacity scaling."
        )
    if operator == "avoid_failed_family":
        return (
            f"Avoid the recently failed {direction} family and choose a different "
            "mechanism under the assigned boundary."
        )
    return (
        f"Explore one new mechanism under {direction}; do not only tune numeric "
        "constants such as width, depth, learning rate, dropout, or batch size."
    )


def _default_rationale(stats: dict[str, Any], summary: dict[str, Any]) -> str:
    direction = str(stats.get("direction_key") or "selected direction")
    visits = int(stats.get("visits") or 0)
    keep_count = int(stats.get("keep_count") or 0)
    active = summary.get("active_directions") or []
    blocked = stats.get("recent_failure_reasons") or []
    return (
        f"{direction} has visits={visits}, keep_count={keep_count}. "
        f"Active directions are {active}. Recent failure reasons are {blocked}. "
        "The runner selected this plan to balance exploration and exploitation."
    )


def _new_plan_id(candidate_id: str, operator: str, direction: str) -> str:
    raw = f"{candidate_id}:{operator}:{direction}:{time.time_ns()}"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"plan-{candidate_id}-{suffix}"


def select_search_plan(
    *,
    candidate_id: str,
    registry_items: list[dict[str, Any]],
    parent_ref: str,
    parent_candidate_id: str | None = None,
) -> SearchPlan:
    summary = summarize_registry(registry_items)
    direction, stats, priority = _choose_direction(summary)
    operator = _operator_for_direction(stats)
    search_mode = "control_tune" if operator == "control_tune" else "mechanism_search"
    if operator == "avoid_failed_family":
        search_mode = "pivot_after_failure"
    elif operator == "pivot_near_miss":
        search_mode = "pivot_after_failure"

    avoid = set(summary.get("active_directions") or [])
    avoid.update(summary.get("blocked_directions") or [])
    avoid.discard(direction)
    if not stats:
        stats = {
            "direction_key": direction,
            "visits": 0,
            "keep_count": 0,
            "discard_count": 0,
            "blocked_count": 0,
            "active_count": 0,
            "recent_failure_reasons": [],
        }

    return SearchPlan(
        plan_id=_new_plan_id(candidate_id, operator, direction),
        parent_candidate_id=parent_candidate_id,
        parent_ref=parent_ref,
        operator=operator,
        target_direction_key=direction,
        intent=_default_intent(search_mode, operator, direction),
        rationale=_default_rationale(stats, summary),
        avoid_direction_keys=sorted(avoid),
        priority=round(priority, 6),
        search_mode=search_mode,
        experiment_brief=(
            "Implement one minimal train.py change that follows this plan and "
            "keeps unrelated settings fixed."
        ),
        expected_mechanism="The diff should change the assigned mechanism, not merely tune capacity.",
        success_interpretation="Lower val_bpb supports the selected mechanism under this parent.",
        failure_interpretation="No improvement suggests pivoting or rejecting this mechanism family.",
        stats={
            "selected_direction": stats,
            "summary": summary,
        },
    )


def complete_plan_from_payload(plan: SearchPlan, payload: dict[str, Any] | None) -> SearchPlan:
    if not isinstance(payload, dict):
        return plan

    allowed = {
        "intent",
        "rationale",
        "experiment_brief",
        "expected_mechanism",
        "success_interpretation",
        "failure_interpretation",
    }
    data = plan.to_dict()
    for key in allowed:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            data[key] = value.strip()[:2000]
    return SearchPlan(**data)


def _looks_like_pure_tuning_text(text: str) -> bool:
    lower = text.lower()
    hits = sum(1 for key in PURE_TUNING_KEYWORDS if key in lower)
    mechanism_words = {
        "attention path",
        "loss term",
        "optimizer update",
        "routing",
        "curriculum",
        "normalization structure",
        "residual path",
        "mechanism",
        "architecture path",
        "computation path",
    }
    has_mechanism = any(word in lower for word in mechanism_words)
    return hits >= 2 and not has_mechanism


def validate_search_plan(plan: SearchPlan) -> tuple[bool, str]:
    if not plan.plan_id:
        return False, "missing_plan_id"
    if plan.operator not in DEFAULT_OPERATORS and plan.operator not in {
        "control_tune",
        "verify_winner",
        "merge_validate",
        "simplify_winner",
    }:
        return False, f"invalid_operator:{plan.operator}"
    if not plan.target_direction_key or plan.target_direction_key == "unknown":
        return False, "missing_target_direction_key"
    if not plan.intent.strip():
        return False, "missing_intent"
    if not plan.rationale.strip():
        return False, "missing_rationale"
    if plan.search_mode == "mechanism_search":
        merged = " ".join(
            [
                plan.target_direction_key,
                plan.intent,
                plan.experiment_brief,
                plan.expected_mechanism,
            ]
        )
        if _looks_like_pure_tuning_text(merged):
            return False, "mechanism_search_plan_looks_like_pure_tuning"
    return True, ""


def plan_prompt_payload(plan: SearchPlan) -> str:
    payload = {
        "hard_constraints": {
            "plan_id": plan.plan_id,
            "parent_candidate_id": plan.parent_candidate_id,
            "parent_ref": plan.parent_ref,
            "operator": plan.operator,
            "target_direction_key": plan.target_direction_key,
            "avoid_direction_keys": plan.avoid_direction_keys,
            "search_mode": plan.search_mode,
        },
        "draft_semantics": {
            "intent": plan.intent,
            "rationale": plan.rationale,
            "experiment_brief": plan.experiment_brief,
            "expected_mechanism": plan.expected_mechanism,
            "success_interpretation": plan.success_interpretation,
            "failure_interpretation": plan.failure_interpretation,
        },
        "stats": plan.stats,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def compute_reward(
    *,
    baseline_before: float,
    val_bpb: float | None,
    final_status: str,
    run_status: str,
    discard_reason: str | None = None,
) -> tuple[float, str]:
    if final_status == "keep" and val_bpb is not None:
        if math.isfinite(baseline_before) and baseline_before > 0:
            return (baseline_before - val_bpb) / baseline_before * 100.0, "relative_bpb_improvement_percent"
        return 1.0, "keep_default"
    if final_status == "discard" and run_status == "completed":
        return -0.2, "completed_not_improved"
    if discard_reason == "repair_out_of_scope":
        return -1.0, "repair_out_of_scope"
    if run_status in {"timeout", "oom", "failed", "crashed"}:
        return -0.8, f"run_status_{run_status}"
    return -0.5, "discard_default"


def search_plan_to_prompt(plan: SearchPlan) -> str:
    return json.dumps(plan.public_dict(), ensure_ascii=False, indent=2)
