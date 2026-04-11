"""metric_judge: 比较 val_bpb，判定 keep/discard"""

from __future__ import annotations


def judge(
    baseline_bpb: float,
    new_bpb: float,
    direction: str = "minimize",
) -> tuple[bool, str]:
    """
    判定新候选是否优于基线。

    Args:
        baseline_bpb: 当前最佳 val_bpb
        new_bpb: 新候选 val_bpb
        direction: minimize（越低越好）或 maximize（越高越好）

    Returns:
        (improved, decision)
    """
    if direction == "minimize":
        improved = new_bpb < baseline_bpb
    else:
        improved = new_bpb > baseline_bpb

    decision = "keep" if improved else "discard"
    return improved, decision
