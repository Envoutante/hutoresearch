"""ClaudeCodeAgent: 调用 Claude Code CLI (claude -p) 生成/修改 train.py"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_text(data: bytes | None) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", errors="replace")


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """Read all .py files from workdir (flat, no subdirs)."""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    return files


@dataclass
class CodeAgentResult:
    """ClaudeCodeAgent 的返回结果"""
    success: bool
    content: str  # train.py 代码内容
    rc: int  # subprocess return code
    stderr: str
    elapsed: float  # seconds
    timed_out: bool
    files: dict[str, str]  # workdir 中的 .py 文件


def _run_subprocess(
    cmd: list[str],
    workdir: Path,
    timeout_sec: int,
) -> tuple[int, str, str, float, bool]:
    """Run command as subprocess with process-group cleanup on timeout.

    Returns (returncode, stdout, stderr, elapsed_sec, timed_out).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env={**os.environ},
        start_new_session=True,
    )
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            stdout_bytes, stderr_bytes = proc.communicate(timeout=5)

    elapsed = time.monotonic() - start
    return (
        proc.returncode if proc.returncode is not None else -1,
        _to_text(stdout_bytes),
        _to_text(stderr_bytes),
        elapsed,
        timed_out,
    )


class ClaudeCodeAgent:
    """Backed by Claude Code CLI (claude -p)."""

    def __init__(
        self,
        model: str = "sonnet",
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
    ):
        self._model = model
        self._timeout_sec = timeout_sec
        self._extra_args = extra_args or []
        self._binary = "claude"

    def _build_result(
        self,
        workdir: Path,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        timed_out: bool,
    ) -> CodeAgentResult:
        """Collect .py files from workdir and build result."""
        files = _collect_py_files(workdir)
        error = None
        if timed_out:
            error = f"Timed out after {elapsed:.0f}s"
        elif returncode != 0 and not files:
            error = f"Exited {returncode}: {stderr[:500]}"

        # 优先从 workdir 读 train.py，否则从 stdout 提取
        content = ""
        if "train.py" in files:
            content = files["train.py"]
        elif stdout.strip():
            content = stdout.strip()

        return CodeAgentResult(
            success=(error is None and bool(content)),
            content=content,
            rc=returncode,
            stderr=stderr,
            elapsed=elapsed,
            timed_out=timed_out,
            files=files,
        )

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--allowed-tools", "Bash Edit Write Read",
            "--add-dir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd

    def _run_subprocess(
        self,
        cmd: list[str],
        workdir: Path,
        timeout_sec: int,
    ) -> tuple[int, str, str, float, bool]:
        return _run_subprocess(cmd, workdir, timeout_sec)

    def _generate_prompt(
        self,
        topic: str,
        exp_plan: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
    ) -> str:
        """生成首次 generate 的 prompt"""
        prompt = f"""你是 AI 研究员，正在对 GPT 训练代码 train.py 进行实验优化。

## 任务
{topic}

## 实验目标
{exp_plan}

## 评估指标
指标名: {metric_key}
方向: minimize（越低越好）

## 计算预算
{compute_budget}

## 当前 train.py 内容摘要
{pkg_hint}

## 优化方向指引
除了超参数（学习率、batch size、权重衰减等）外，还可以从以下维度寻找突破：
1. **模型结构**：层数、embedding 维度、attention head 数、KV head 数、window pattern
2. **激活函数**：SwiGLU、GeGLU、ReLU、GeLU 等
3. **归一化策略**：RMSNorm、LayerNorm、Pre-Norm、Post-Norm
4. **训练范式**：初始化策略、梯度裁剪、warmup 策略、learning rate schedule
5. **混合精度与算子**：BF16 vs FP16、FlashAttention 版本、kernel 实现

请充分分析当前代码和上述维度，选择一个最有潜力的方向进行改进。

## 额外指导
{extra_guidance}

## 输出要求
请直接输出完整的新 train.py 代码。不要解释，直接给出代码。
"""
        return prompt

    def _refine_prompt(
        self,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
    ) -> str:
        """生成 refine 的 prompt"""
        summaries_text = "\n".join(run_summaries[-10:]) if run_summaries else "无历史记录"
        files_text = ""
        for name, content in current_files.items():
            files_text += f"\n=== {name} ===\n{content[:3000]}"

        prompt = f"""你是 AI 研究员，正在对 train.py 进行迭代优化。

## 任务
{topic}

## 指标
指标名: {metric_key}
方向: {metric_direction}

## 历史运行摘要（按时间顺序，近似到远）
{summaries_text}

请**充分分析**以上历史实验结果：
- 哪些改动带来了提升？提升了多少？
- 哪些改动没有效果，甚至变差了？
- 被 discard 的实验中，有哪些思路值得重新尝试（比如参数设置更合理）？
- 哪些优化方向还没有被探索过？

在分析的基础上，选择一个最有潜力的方向进行改进。

## 当前文件
{files_text}

## 优化方向指引
除了超参数外，还可以从以下维度寻找突破：
1. **模型结构**：层数、embedding 维度、attention head 数、KV head 数、window pattern
2. **激活函数**：SwiGLU、GeGLU、ReLU、GeLU 等
3. **归一化策略**：RMSNorm、LayerNorm、Pre-Norm、Post-Norm
4. **训练范式**：初始化策略、梯度裁剪、warmup 策略、learning rate schedule
5. **混合精度与算子**：BF16 vs FP16、FlashAttention 版本、kernel 实现

## 额外提示
{extra_hints}

## 输出要求
请直接输出完整的新 train.py 代码。不要解释，直接给出代码。
"""
        return prompt

    def _repair_prompt(
        self,
        files: dict[str, str],
        issues: str,
    ) -> str:
        """生成 repair 的 prompt"""
        files_text = ""
        for name, content in files.items():
            files_text += f"\n=== {name} ===\n{content[:3000]}"

        prompt = f"""你是 AI 研究员，正在修复 train.py 的问题。

## 当前文件
{files_text}

## 问题描述
{issues}

## 输出要求
请直接输出修复后的完整 train.py 代码。不要解释，直接给出代码。
"""
        return prompt

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        prompt = self._generate_prompt(
            topic, exp_plan, metric_key, pkg_hint, compute_budget, extra_guidance,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._timeout_sec,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        prompt = self._refine_prompt(
            current_files, run_summaries, metric_key, metric_direction,
            topic, extra_hints,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._timeout_sec,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        prompt = self._repair_prompt(files, issues)
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._timeout_sec,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)
