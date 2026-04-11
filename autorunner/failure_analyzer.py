"""failure_analyzer: 失败分类与修复策略映射"""

from __future__ import annotations

from dataclasses import dataclass


# 失败类型 -> 修复策略
FAILURE_RETRY = {
    "dependency": "retry",     # 依赖缺失，修复后重试
    "runtime": "retry",        # 运行报错，修复后重试
    "timeout": "skip",         # 超时不重试
    "metric_anomaly": "skip",  # NaN/Inf 不重试
}


@dataclass
class FailureAnalysis:
    failure_type: str
    should_retry: bool
    repair_strategy: str  # 修复建议


def analyze(failure_type: str, log_tail: str = "") -> FailureAnalysis:
    """
    分析失败类型，返回是否应该重试及修复策略。

    Args:
        failure_type: 失败类型（timeout/dependency/runtime/metric_anomaly/none）
        log_tail: 错误日志尾部

    Returns:
        FailureAnalysis
    """
    if failure_type not in FAILURE_RETRY:
        failure_type = "runtime"

    should_retry = FAILURE_RETRY.get(failure_type, "skip") == "retry"

    # 生成修复建议
    if failure_type == "dependency":
        strategy = "检查 ImportError/ModuleNotFoundError，安装缺失依赖或修复 import 语句"
    elif failure_type == "runtime":
        strategy = "检查 Traceback，修复 Python 运行时错误（类型错误、属性错误等）"
    elif failure_type == "timeout":
        strategy = "训练超时，考虑减少模型规模/批量大小/步数，或增加 time_budget"
    elif failure_type == "metric_anomaly":
        strategy = "检测到 NaN/Inf，检查梯度爆炸、学习率过大、初始化问题"
    else:
        strategy = "未知错误"

    return FailureAnalysis(
        failure_type=failure_type,
        should_retry=should_retry,
        repair_strategy=strategy,
    )
