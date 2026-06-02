"""Deterministic local-parameter tuning helpers."""

from __future__ import annotations

import ast
import json
import math
from typing import Any


def strip_inline_comment(value: Any) -> str:
    return str(value or "").split("#", 1)[0].strip()


def parse_tuning_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else value
    if isinstance(value, (list, tuple)):
        return tuple(parse_tuning_value(x) for x in value)

    text = strip_inline_comment(value)
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return float(text)
        except ValueError:
            return text
    return parse_tuning_value(parsed)


def canonical_tuning_value(value: Any) -> Any:
    parsed = parse_tuning_value(value)
    if isinstance(parsed, list):
        return tuple(canonical_tuning_value(x) for x in parsed)
    if isinstance(parsed, tuple):
        return tuple(canonical_tuning_value(x) for x in parsed)
    if isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed


def tuning_values_equal(left: Any, right: Any) -> bool:
    lval = canonical_tuning_value(left)
    rval = canonical_tuning_value(right)
    if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
        return math.isclose(float(lval), float(rval), rel_tol=1e-12, abs_tol=0.0)
    return lval == rval


def tuning_value_key(value: Any) -> str:
    return json.dumps(
        canonical_tuning_value(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def tuning_value_to_source(value: Any) -> str:
    value = canonical_tuning_value(value)
    if isinstance(value, tuple):
        inner = ", ".join(tuning_value_to_source(x) for x in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, float):
        return f"{value:.12g}"
    return repr(value) if isinstance(value, str) else str(value)


def _as_sequence(value: Any) -> tuple[Any, ...] | None:
    parsed = canonical_tuning_value(value)
    return parsed if isinstance(parsed, tuple) else None


def _replace_sequence_value(full_value: Any, index: int, leaf_value: Any) -> tuple[Any, ...]:
    seq = _as_sequence(full_value)
    if seq is None:
        return (canonical_tuning_value(leaf_value),)
    if index < 0 or index >= len(seq):
        return seq
    out = list(seq)
    out[index] = canonical_tuning_value(leaf_value)
    return tuple(out)


def infer_control_index(
    *,
    control_name: str,
    actual_value: Any,
    candidate_values: list[Any],
    requested_index: Any = None,
) -> int | None:
    try:
        explicit = int(requested_index)
    except (TypeError, ValueError):
        explicit = None
    seq = _as_sequence(actual_value)
    if explicit is not None and seq is not None and 0 <= explicit < len(seq):
        return explicit

    upper_name = control_name.upper()
    if seq is not None and len(seq) >= 2 and ("BETA" in upper_name or upper_name.endswith("BETAS")):
        return 1

    differing_indexes: set[int] = set()
    for raw in candidate_values:
        cand = _as_sequence(raw)
        if cand is None or seq is None or len(cand) != len(seq):
            continue
        diffs = [
            i for i, (left, right) in enumerate(zip(seq, cand, strict=True))
            if not tuning_values_equal(left, right)
        ]
        if len(diffs) == 1:
            differing_indexes.add(diffs[0])
    if len(differing_indexes) == 1:
        return next(iter(differing_indexes))
    return None


def _numeric_leaf(value: Any, control_index: int | None = None) -> float | None:
    parsed = canonical_tuning_value(value)
    if control_index is not None and isinstance(parsed, tuple):
        if 0 <= control_index < len(parsed):
            parsed = parsed[control_index]
    if isinstance(parsed, (int, float)) and math.isfinite(float(parsed)):
        return float(parsed)
    return None


def _round_like(value: float, template: float) -> float | int:
    if float(template).is_integer() and abs(value) >= 2:
        return int(round(value))
    return round(value, 8)


def propose_candidate_values(
    *,
    actual_value: Any,
    control_index: int | None,
    previous_trials: list[dict[str, Any]],
) -> list[Any]:
    current = _numeric_leaf(actual_value, control_index)
    if current is None:
        return []

    if abs(current) >= 2:
        step = max(1.0, abs(current) * 0.125)
        raw_values = [current - step, current + step, current + 2 * step]
    elif 0.0 < current < 1.0:
        step = max(0.0001, min(0.05, current * 0.0064935065))
        raw_values = [current - step, current + step, current + 2 * step]
    else:
        step = max(0.0001, abs(current) * 0.25)
        raw_values = [current - step, current + step, current + 2 * step]

    used = {tuning_value_key(x.get("selected_value")) for x in previous_trials}
    out: list[Any] = []
    for raw in raw_values:
        value = _round_like(raw, current)
        if 0.0 < current < 1.0:
            value = min(0.999999, max(0.000001, float(value)))
        full = (
            _replace_sequence_value(actual_value, control_index, value)
            if control_index is not None
            else canonical_tuning_value(value)
        )
        source = tuning_value_to_source(full)
        if tuning_value_key(source) in used:
            continue
        if source not in out:
            out.append(source)
        if len(out) >= 3:
            break
    return out


def normalize_tuning_plan_with_policy(
    *,
    tuning_plan: dict[str, Any],
    parent_assignments: dict[str, str],
    previous_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    plan = dict(tuning_plan or {})
    control_name = str(plan.get("control_name") or "").strip()
    if not control_name:
        return plan

    actual_assignment = parent_assignments.get(control_name)
    if actual_assignment is None:
        return plan

    raw_candidates = plan.get("candidate_values")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    actual_value = canonical_tuning_value(actual_assignment)
    control_index = infer_control_index(
        control_name=control_name,
        actual_value=actual_value,
        candidate_values=raw_candidates,
        requested_index=plan.get("control_index"),
    )

    normalized_candidates: list[str] = []
    for raw in raw_candidates:
        parsed = canonical_tuning_value(raw)
        if control_index is not None:
            seq = _as_sequence(parsed)
            leaf = (
                seq[control_index]
                if seq is not None and 0 <= control_index < len(seq)
                else parsed
            )
            parsed = _replace_sequence_value(actual_value, control_index, leaf)
        source = tuning_value_to_source(parsed)
        if source not in normalized_candidates:
            normalized_candidates.append(source)
        if len(normalized_candidates) >= 3:
            break

    if not normalized_candidates:
        normalized_candidates = propose_candidate_values(
            actual_value=actual_value,
            control_index=control_index,
            previous_trials=previous_trials,
        )

    used = {
        tuning_value_key(trial.get("selected_value"))
        for trial in previous_trials
        if trial.get("selected_value") is not None
    }
    selected_value = normalized_candidates[0] if normalized_candidates else None
    for value in normalized_candidates:
        if tuning_value_key(value) not in used:
            selected_value = value
            break

    plan["control_name"] = control_name
    plan["control_index"] = control_index
    plan["current_value"] = tuning_value_to_source(actual_value)
    plan["candidate_values"] = normalized_candidates
    plan["selected_value"] = selected_value
    if control_index is not None:
        seq = _as_sequence(selected_value)
        if seq is not None and 0 <= control_index < len(seq):
            plan["selected_leaf_value"] = tuning_value_to_source(seq[control_index])
    return plan
