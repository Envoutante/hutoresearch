# Parallel Runner 重构蓝图（可执行）

## 1. 目标

将当前串行 `loop_runner` 升级为并行流水线，满足：

1. Generator 生成多个候选实验，并基于 `train.py` 副本运行，避免修改主工作区文件。
2. 调度器维护候选队列：
   - GPU 空闲则拉起实验；
   - 队列空位则唤醒 Generator 继续产出；
   - 避免重复生成已失败方向。

## 2. 目录与产物约定

建议新增目录：

- `autorunner/candidates/`：候选工作目录根目录
- `autorunner/candidates/cand-000123/`：单候选目录
  - `train.py`
  - `meta.json`
  - `run.log`
  - `llm_refine.log`
  - `llm_repair.log`（可选）

建议新增状态文件：

- `autorunner/artifacts/parallel_queue.jsonl`：任务状态流（append-only）
- `autorunner/artifacts/failure_directions.json`：失败方向记忆库
- `autorunner/artifacts/parallel_state.json`：调度器恢复点（可选）

## 3. 数据模型（建议 dataclass）

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CandidateStatus = Literal[
    "queued", "running", "repairing", "completed", "discarded", "rejected"
]

@dataclass
class CandidateTask:
    candidate_id: str
    parent_ref: str  # best commit/hash 或 "baseline"
    workdir: Path
    train_py_path: Path
    refine_description: str
    direction_fingerprint: str
    created_at: float
    status: CandidateStatus = "queued"
    gpu_id: int | None = None
    attempt: int = 0
    repair_attempt: int = 0

@dataclass
class GpuSlot:
    gpu_id: int
    busy: bool = False
    candidate_id: str | None = None
    pid: int | None = None

@dataclass
class FailureDirection:
    fingerprint: str
    description: str
    reason: str
    first_seen_ts: float
    count: int = 1
```

## 4. 新增模块建议

- `autorunner/parallel_runner.py`：并行 orchestrator 主入口
- `autorunner/candidate_store.py`：候选目录创建、meta 读写、状态持久化
- `autorunner/gpu_scheduler.py`：GPU 空闲探测与 slot 管理
- `autorunner/direction_memory.py`：失败方向记忆与去重判定

> 先不拆也可以：第一版可以先都写在 `parallel_runner.py`，稳定后再拆。

## 5. 与现有模块的改造边界

### 5.1 `claude_code_agent.py`

保持接口风格，新增可选方法：

```python
def refine_in_workspace(
    self,
    *,
    candidate_workdir: Path,
    run_summaries: list[str],
    failure_directions: list[str],
    topic: str,
    timeout_sec: int | None = None,
) -> CodeAgentResult:
    ...
```

说明：
- 本质上仍调用现有 `refine`，但 `workdir` 指向候选目录；
- `failure_directions` 注入 prompt，要求避开失败方向。

### 5.2 `experiment_executor.py`

在 `run()` 增加两个参数：

```python
def run(
    train_py_path: Path,
    time_budget: int = 600,
    run_log_path: Path | None = None,
    on_process_started: Callable[[int], None] | None = None,
    env_overrides: dict[str, str] | None = None,
    label: str | None = None,
) -> ExperimentResult:
    ...
```

说明：
- `env_overrides` 用于注入 `CUDA_VISIBLE_DEVICES=<gpu_id>`；
- `label` 用于日志打点，定位候选。

### 5.3 `loop_runner.py`

建议保留原串行入口不删，新增并行入口脚本：
- `python -m autorunner.parallel_runner --max-running 2 --queue-size 6 ...`

## 6. 核心函数签名（第一版最小可运行）

```python
# parallel_runner.py

def run_parallel_loop(
    max_total_runs: int,
    queue_capacity: int,
    max_running: int,
    time_budget: int,
    topic: str | None,
    poll_interval_sec: float = 2.0,
) -> None:
    ...


def maybe_fill_queue(
    *,
    target_size: int,
    candidate_root: Path,
    base_train_path: Path,
    run_summaries: list[str],
    failure_memory: FailureDirectionMemory,
    agent: ClaudeCodeAgent,
    topic: str,
) -> int:
    """当队列未满时生成候选，返回新增数量。"""


def launch_runnable_candidates(
    *,
    pending: list[CandidateTask],
    gpu_slots: list[GpuSlot],
    time_budget: int,
) -> list[CandidateTask]:
    """给空闲 GPU 分发任务，返回已启动任务列表。"""


def collect_finished_candidates(
    *,
    running: dict[str, CandidateTask],
    baseline_bpb: float,
    failure_memory: FailureDirectionMemory,
) -> tuple[list[CandidateTask], float]:
    """回收已完成任务并更新 baseline。"""


def evaluate_candidate_result(
    task: CandidateTask,
    exp_result: ExperimentResult,
    baseline_bpb: float,
) -> tuple[str, bool]:
    """返回 decision(keep/discard), improved。"""
```

## 7. 调度主循环伪代码

```python
while not stop_condition:
    # A. 资源探测
    gpu_slots = probe_gpu_slots()

    # B. 队列补货（低水位触发）
    if queued_count < low_watermark:
        maybe_fill_queue(...)

    # C. 出队运行（空闲 GPU 触发）
    launch_runnable_candidates(...)

    # D. 回收完成任务
    done, baseline_bpb = collect_finished_candidates(...)

    # E. 聚合落盘（单写者）
    for task in done:
        append_results_tsv(task)
        write_iter_artifact(task)
        update_failure_memory_if_needed(task)

    sleep(poll_interval)
```

## 8. 候选生成流程（副本隔离）

1. 从基线 `train.py` 复制到新候选目录。
2. 调用 `agent.refine(..., workdir=candidate_workdir)`。
3. 从回复提取 `refine_description`。
4. 计算 `direction_fingerprint`：
   - `normalized(description)` +
   - 关键参数 diff（如 `DEPTH/ASPECT_RATIO/HEAD_DIM/WINDOW_PATTERN/DEVICE_BATCH_SIZE/LR`）哈希。
5. 若与失败方向相似度超阈值（或命中完全指纹），直接标记 `rejected`，并生成新候选替换。
6. 入队。

## 9. 失败方向记忆（防重复）

### 9.1 记忆来源

候选满足任一条件时写入失败记忆：
- 运行崩溃且 repair 失败；
- repair 越界（改了优化参数）被拦截；
- 指标显著恶化（可选阈值）。

### 9.2 记忆结构

`failure_directions.json` 维护：

```json
{
  "items": [
    {
      "fingerprint": "sha1:...",
      "description": "增大模型宽度并减小batch",
      "reason": "repair_out_of_scope",
      "count": 3,
      "first_seen_ts": 1710000000.0
    }
  ]
}
```

### 9.3 过滤策略

- 硬过滤：fingerprint 完全命中 -> 直接拒绝。
- 软过滤：description 语义相似（简化版可用 token overlap）-> 提示 generator 避开。

## 10. Repair 策略在并行中的继承

沿用你现有约束：
- repair 仅允许最小可运行修复；
- 禁止改优化/结构参数；
- 越界即判本候选 `discarded`。

并行模式下，这些检查在候选目录执行，不影响其他候选。

## 11. 结果汇总与兼容性

建议 `results.tsv` 扩展列（向后兼容可追加到末尾）：

- `candidate_id`
- `parent_ref`
- `gpu_id`
- `fingerprint`
- `discard_reason`

如果暂不改 `results.tsv`，可先写到 `iter-*.json`，后续再汇总。

## 12. 分阶段实施（推荐）

### Phase 1（1-2 天）
- 新增 `parallel_runner.py`，实现：
  - 候选副本生成
  - 队列补货
  - 单 GPU 运行 + 单写者聚合
- 保持 `results.tsv` 原格式，先跑通。

### Phase 2（1-2 天）
- 多 GPU 调度（slot 模型）
- `CUDA_VISIBLE_DEVICES` 注入
- 更完善状态恢复（重启续跑）

### Phase 3（1-2 天）
- 失败方向记忆库
- 指纹去重与 prompt 反哺
- 指标与吞吐监控

## 13. 关键工程建议

1. 并行模式不要做 `git commit` 作为候选追踪主键，改用 `candidate_id`。
2. 统一由 aggregator 写 `results.tsv`，避免并发写冲突。
3. 每个候选独立 `run.log`，不要共用全局 `run.log`。
4. 先保证正确性再拉高并发；`max_running` 默认从 1 起步验证。

## 14. 可直接执行的下一步

1. 创建 `autorunner/parallel_runner.py` 骨架与参数解析。
2. 从 `loop_runner.py` 抽出可复用函数：
   - description 提取
   - results.tsv 追加
   - iter artifact 写入
3. 在 `experiment_executor.run()` 增加 `env_overrides`。
4. 用 `--max-running 1 --queue-size 3` 先跑通，然后再开多卡。
