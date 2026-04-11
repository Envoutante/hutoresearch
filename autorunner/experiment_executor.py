"""experiment_executor: 运行 train.py 并捕获结果（安静模式）"""

from __future__ import annotations

import re
import subprocess
import threading
import time
import os
import signal
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    """单次实验结果"""
    status: str  # 'completed' | 'timeout' | 'crashed'
    val_bpb: float | None
    peak_vram_mb: float | None
    training_seconds: float | None
    total_seconds: float | None
    mfu_percent: float | None
    total_tokens_M: float | None
    num_steps: int | None
    num_params_M: float | None
    depth: int | None
    failure_type: str | None  # 'none' | 'timeout' | 'dependency' | 'runtime' | 'metric_anomaly'
    log_tail: str
    runtime_seconds: float


def _parse_log(log_content: str) -> dict:
    """从 run.log 解析实验结果"""
    result = {}

    # val_bpb
    m = re.search(r"^val_bpb:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["val_bpb"] = float(m.group(1))

    # peak_vram_mb
    m = re.search(r"^peak_vram_mb:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["peak_vram_mb"] = float(m.group(1))

    # training_seconds
    m = re.search(r"^training_seconds:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["training_seconds"] = float(m.group(1))

    # total_seconds
    m = re.search(r"^total_seconds:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["total_seconds"] = float(m.group(1))

    # mfu_percent
    m = re.search(r"^mfu_percent:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["mfu_percent"] = float(m.group(1))

    # total_tokens_M
    m = re.search(r"^total_tokens_M:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["total_tokens_M"] = float(m.group(1))

    # num_steps
    m = re.search(r"^num_steps:\s*(\d+)", log_content, re.MULTILINE)
    if m:
        result["num_steps"] = int(m.group(1))

    # num_params_M
    m = re.search(r"^num_params_M:\s*([\d.]+)", log_content, re.MULTILINE)
    if m:
        result["num_params_M"] = float(m.group(1))

    # depth
    m = re.search(r"^depth:\s*(\d+)", log_content, re.MULTILINE)
    if m:
        result["depth"] = int(m.group(1))

    return result


def _classify_failure(log_content: str) -> str:
    """分类失败类型"""
    if "TimeoutExpired" in log_content or "timed out" in log_content.lower():
        return "timeout"
    if "ModuleNotFoundError" in log_content or "ImportError" in log_content:
        return "dependency"
    if "out of memory" in log_content.lower() or "OOM" in log_content:
        return "runtime"
    error_patterns = [
        "Traceback (most recent call last)",
        "RuntimeError",
        "ValueError",
        "TypeError",
    ]
    for pattern in error_patterns:
        if pattern in log_content:
            return "runtime"
    if "nan" in log_content.lower() or "inf" in log_content.lower():
        return "metric_anomaly"
    if "val_bpb" not in log_content:
        return "runtime"
    return "none"


def run(
    train_py_path: Path,
    time_budget: int = 600,
    run_log_path: Path | None = None,
) -> ExperimentResult:
    """
    运行 train.py，实时写入 run.log。

    Args:
        train_py_path: train.py 文件路径
        time_budget: 训练时间预算（秒）
        run_log_path: run.log 写入路径（默认 train.py 同目录下 run.log）
    """
    if run_log_path is None:
        run_log_path = train_py_path.parent / "run.log"

    start = time.monotonic()
    runtime_seconds = 0.0

    # 实时写入 run.log
    timed_out = False
    killed = threading.Event()

    def kill_after_timeout():
        nonlocal timed_out
        killed.wait(timeout=time_budget)
        if not killed.is_set():
            timed_out = True
            # uv 可能会再派生 python 子进程；超时时需要杀整个进程组
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            except Exception:
                # 兜底：至少确保父进程被终止
                process.kill()

    with run_log_path.open("w", buffering=1) as log_file:
        # 强制 Python 子进程无缓冲输出，避免 run.log 长时间为空
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            ["uv", "run", "python", "-u", "train.py"],
            cwd=train_py_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )

        # 在“Run experiment”下一行打印当前实验进程 PID，便于排障
        print(f"       pid: {process.pid}")

        killer = threading.Thread(target=kill_after_timeout, daemon=True)
        killer.start()

        full_lines = []
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                full_lines.append(line)
        except Exception:
            pass

        process.wait()
        killer.join(timeout=1)
        killed.set()
        rc = process.returncode

    full_log = "".join(full_lines)
    runtime_seconds = time.monotonic() - start

    # 解析日志
    parsed = _parse_log(full_log)

    # 判定状态
    if timed_out or rc == -1:
        status = "timeout"
        failure_type = "timeout"
        val_bpb = None
    elif rc != 0:
        status = "crashed"
        failure_type = _classify_failure(full_log)
        val_bpb = None
    elif parsed.get("val_bpb") is None:
        status = "crashed"
        failure_type = "metric_anomaly"
        val_bpb = None
    else:
        status = "completed"
        failure_type = "none"
        val_bpb = parsed["val_bpb"]

    # 取最后 50 行作为 log_tail
    lines = full_log.splitlines()
    log_tail = "\n".join(lines[-50:]) if len(lines) > 50 else full_log

    return ExperimentResult(
        status=status,
        val_bpb=val_bpb,
        peak_vram_mb=parsed.get("peak_vram_mb"),
        training_seconds=parsed.get("training_seconds"),
        total_seconds=parsed.get("total_seconds"),
        mfu_percent=parsed.get("mfu_percent"),
        total_tokens_M=parsed.get("total_tokens_M"),
        num_steps=parsed.get("num_steps"),
        num_params_M=parsed.get("num_params_M"),
        depth=parsed.get("depth"),
        failure_type=failure_type,
        log_tail=log_tail,
        runtime_seconds=runtime_seconds,
    )
