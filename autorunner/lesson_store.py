"""lesson_store: lessons JSONL 读写与检索"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACTS_DIR = Path("/mount/disk1/rl-hyr/autoresearch/autorunner/artifacts")
LESSONS_FILE = ARTIFACTS_DIR / "lessons.jsonl"


def _extract_lesson_type(notes: str) -> str:
    """从 notes 中提取 lesson 类型标签"""
    # 简单规则：识别常见的改动类型关键词
    lower = notes.lower()
    if "lr" in lower or "learning" in lower:
        return "learning_rate"
    if "depth" in lower:
        return "architecture_depth"
    if "head" in lower:
        return "architecture_head"
    if "batch" in lower:
        return "batch_size"
    if "embed" in lower:
        return "embedding"
    if "activation" in lower:
        return "activation"
    if "loss" in lower:
        return "loss"
    if "init" in lower:
        return "initialization"
    if "optimizer" in lower:
        return "optimizer"
    if "dropout" in lower or "reg" in lower:
        return "regularization"
    return "general"


def add_lesson(
    iteration: int,
    decision: str,
    val_bpb: float | None,
    notes: str,
    improved: bool,
):
    """
    从实验结果生成一条 lesson 并追加到 lessons.jsonl。

    Args:
        iteration: 实验轮次
        decision: keep/discard
        val_bpb: 实验 val_bpb
        notes: 实验 notes
        improved: 是否改善了 val_bpb
    """
    LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    lesson_type = _extract_lesson_type(notes)

    # 生成简短 lesson
    if decision == "keep":
        text = f"iter{iteration}: val_bpb={val_bpb:.4f}, keep - {notes}"
    elif decision == "discard":
        text = f"iter{iteration}: val_bpb={val_bpb:.4f}, discard - {notes}"
    else:
        text = f"iter{iteration}: {decision} - {notes}"

    entry = {
        "iteration": iteration,
        "lesson_type": lesson_type,
        "text": text,
        "decision": decision,
        "val_bpb": val_bpb,
        "improved": improved,
    }

    with LESSONS_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_lessons(k: int = 5) -> list[dict[str, Any]]:
    """
    获取最近 k 条 lessons。

    Returns:
        list of lesson entries
    """
    if not LESSONS_FILE.exists():
        return []

    lessons = []
    for line in LESSONS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            lessons.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return lessons[-k:]


def format_lessons_for_prompt(lessons: list[dict[str, Any]]) -> str:
    """将 lessons 格式化为可在 prompt 中显示的字符串"""
    if not lessons:
        return "无历史 lesson。"

    lines = []
    for l in lessons:
        lines.append(f"- [{l['lesson_type']}] {l['text']}")

    return "\n".join(lines)
