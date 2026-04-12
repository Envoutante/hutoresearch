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
    """读取 workdir 下所有顶层 .py 文件（不递归子目录）。"""
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
    content: str  # Claude CLI 的文本回复（stdout）
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
        elif returncode != 0:
            error = f"Exited {returncode}: {stderr[:500]}"

        # 日志内容记录 Claude 文本回复，代码改动通过 files['train.py'] 读取
        content = stdout.strip()
        has_train_code = bool(files.get("train.py", "").strip())
        has_reply = bool(content)

        return CodeAgentResult(
            success=(error is None and (has_train_code or has_reply)),
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

## 约束与取舍原则
1. VRAM 是软约束：为了获得有意义的 {metric_key} 改善，允许小幅增加显存，但严禁显存占用急剧膨胀。
2. 简洁性优先：在其他条件相同的情况下，优先更简单的改动。
3. 复杂性与收益权衡：如果改动明显增加复杂性但收益很小（例如约 0.001 的改善却引入大量 hack 代码），通常不应采用。
4. 简化奖励：若删除代码后取得相同或更好结果，应优先保留该方案；即便指标几乎不变，但代码明显更简单，也应倾向保留。

## 想法耗尽时的探索策略
如果你感觉可尝试的想法变少，不要停下，请继续主动探索：
1. 回看代码中引用的论文与实现线索，提取可落地到当前 train.py 的改动点。
2. 重新审阅已给出的上下文与历史结果，寻找未被充分尝试的角度。
3. 组合此前“接近成功”的思路做小步重组。
4. 在可控风险下尝试更激进的架构改动，并保持代码可运行与可评估。

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

## 执行要求（必须遵守）
1. 使用 Read 工具读取 train.py 和 results.tsv。
2. 使用 Edit/Write 工具直接修改 train.py（原地修改，不新建同义副本）。
3. 可用 Bash 工具做轻量自检（例如 python -m py_compile train.py）。
4. 不要只在对话里返回代码块；最终结果应体现在文件系统中的 train.py 里。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述本轮实验改动，英文，5-18词，不含制表符>
仅使用纯文本，不要使用任何 Markdown 语法符号（例如 #、-、*、```）。
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

## 约束与取舍原则
1. VRAM 是软约束：为了获得有意义的 {metric_key} 改善，允许小幅增加显存，但严禁显存占用急剧膨胀。
2. 简洁性优先：在其他条件相同的情况下，优先更简单的改动。
3. 复杂性与收益权衡：如果改动明显增加复杂性但收益很小（例如约 0.001 的改善却引入大量 hack 代码），通常不应采用。
4. 简化奖励：若删除代码后取得相同或更好结果，应优先保留该方案；即便指标几乎不变，但代码明显更简单，也应倾向保留。

## 想法耗尽时的探索策略
如果你感觉可尝试的想法变少，不要停下，请继续主动探索：
1. 回看代码中引用的论文与实现线索，提取可落地到当前 train.py 的改动点。
2. 重新审阅已给出的上下文与历史结果，寻找未被充分尝试的角度。
3. 组合此前“接近成功”的思路做小步重组。
4. 在可控风险下尝试更激进的架构改动，并保持代码可运行与可评估。

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

## 执行要求（必须遵守）
1. 使用 Read 工具读取 train.py。
2. 使用 Edit/Write 工具直接修改 train.py（原地修改，不新建同义副本）。
3. 可用 Bash 工具做轻量自检（例如 python -m py_compile train.py）。
4. 不要只在对话里返回代码块；最终结果应体现在文件系统中的 train.py 里。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述本轮实验改动，英文，5-18词，不含制表符>
仅使用纯文本，不要使用任何 Markdown 语法符号（例如 #、-、*、```）。
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

## 执行要求（必须遵守）
1. 使用 Read 工具读取 train.py。
2. 根据问题描述定位 bug，并使用 Edit/Write 工具直接修复 train.py。
3. 可用 Bash 工具做轻量自检（例如 python -m py_compile train.py）。
4. 不要只在对话里返回代码块；最终结果应体现在文件系统中的 train.py 里。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述修复动作，英文，5-18词，不含制表符>
仅使用纯文本，不要使用任何 Markdown 语法符号（例如 #、-、*、```）。
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
