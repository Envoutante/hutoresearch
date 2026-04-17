"""ClaudeCodeAgent: 调用 Claude Code CLI (claude -p) 生成/修改 train.py"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text

from autorunner.terminal_box import AgentOutputBox, StreamRenderConfig


CHINESE_THINKING_SYSTEM_PROMPT = (
    "请在整个会话中使用中文进行分析与思考。"
    "如果任务要求某个输出字段有固定格式（例如 DESCRIPTION），请严格遵守该格式要求。"
)


@dataclass
class StreamEvent:
    type: str
    data: dict[str, Any]


class Emitter:
    @staticmethod
    def tool_call(
        name: str,
        args: dict[str, Any],
        tool_id: str = "",
        *,
        index: int | None = None,
    ) -> StreamEvent:
        payload: dict[str, Any] = {
            "type": "tool_call",
            "name": name,
            "args": args,
            "id": tool_id,
        }
        if index is not None:
            payload["index"] = index
        return StreamEvent(type="tool_call", data=payload)

    @staticmethod
    def tool_result(
        name: str,
        content: str,
        success: bool,
        *,
        tool_id: str = "",
    ) -> StreamEvent:
        return StreamEvent(
            type="tool_result",
            data={
                "type": "tool_result",
                "name": name,
                "content": content,
                "success": success,
                "id": tool_id,
            },
        )


def normalize_raw_messages(raw: dict[str, Any]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    kind = raw.get("type")

    if kind == "stream_event":
        event = raw.get("event") or {}
        if event.get("type") == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = event.get("index")
                events.append(
                    Emitter.tool_call(
                        name=str(block.get("name") or "unknown"),
                        args=(block.get("input") or {}),
                        tool_id=str(block.get("id") or ""),
                        index=idx if isinstance(idx, int) else None,
                    )
                )
        return events

    if kind == "assistant":
        message = raw.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                events.append(
                    Emitter.tool_call(
                        name=str(block.get("name") or "unknown"),
                        args=(block.get("input") or {}),
                        tool_id=str(block.get("id") or ""),
                    )
                )
                continue
            if block_type == "tool_result":
                content = block.get("content")
                if isinstance(content, str):
                    rendered = content
                else:
                    rendered = str(content or "")
                events.append(
                    Emitter.tool_result(
                        name=str(block.get("name") or "unknown"),
                        content=rendered,
                        success=not bool(block.get("is_error", False)),
                        tool_id=str(block.get("tool_use_id") or ""),
                    )
                )
        return events

    if kind == "user":
        message = raw.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if isinstance(content, str):
                rendered = content
            else:
                rendered = str(content or "")
            events.append(
                Emitter.tool_result(
                    name=str(block.get("name") or "unknown"),
                    content=rendered,
                    success=not bool(block.get("is_error", False)),
                    tool_id=str(block.get("tool_use_id") or ""),
                )
            )
        return events

    return events


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


def _extract_content_from_stream_json(stdout: str) -> str:
    """Extract final assistant text from stream-json output lines."""
    final_result = ""
    text_parts: list[str] = []

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "result":
            final_result = (obj.get("result") or "").strip()
            continue

        if obj.get("type") != "stream_event":
            continue

        event = obj.get("event") or {}
        if event.get("type") != "content_block_delta":
            continue

        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            text_parts.append(delta.get("text") or "")

    if final_result:
        return final_result

    return "".join(text_parts).strip()


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
    render_title: str = "Claude Agent Output",
    stream_output: bool = True,
    transient_output: bool = False,
) -> tuple[int, str, str, float, bool]:
    """Run command as subprocess with process-group cleanup on timeout.

    Returns (returncode, stdout, stderr, elapsed_sec, timed_out).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    artifacts_dir = workdir / "autorunner" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (
        render_title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    live_stream_log = artifacts_dir / f"live_stream_{safe_title}_{int(start)}.jsonl"

    # 调用 Agent
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env={**os.environ},
        start_new_session=True,
        text=True,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    tool_index_to_id: dict[int, str] = {}
    tool_call_state: dict[str, dict[str, bool]] = {}
    tool_call_meta: dict[str, dict[str, Any]] = {}
    thinking_buffer: list[str] = []
    current_assistant_stream_id: str | None = None

    def _format_tool_compact(name: str, args: dict[str, Any] | None) -> str:
        if not args:
            return name
        try:
            compact = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            compact = str(args)
        if len(compact) > 120:
            compact = compact[:117] + "..."
        return f"{name} {compact}"

    def _is_success(content: str, success_flag: bool) -> bool:
        if not success_flag:
            return False
        lowered = content.lower()
        return (
            "<tool_use_error>" not in lowered and "inputvalidationerror" not in lowered
        )

    def _render_tool_call_line(
        name: str, args: dict[str, Any], tr: dict[str, Any] | None
    ) -> Text:
        is_task = name.lower() == "task"
        if tr is not None:
            content = str(tr.get("content") or "")
            if _is_success(content, bool(tr.get("success", True))):
                style = "bold green"
                indicator = "✓"
            else:
                style = "bold red"
                indicator = "✗"
        else:
            style = "bold cyan" if is_task else "bold yellow"
            indicator = "▶"

        tool_text = Text()
        tool_text.append(f"{indicator} ", style=style)
        tool_text.append(_format_tool_compact(name, args), style=style)
        return tool_text

    def _format_tool_result_compact(
        name: str,
        content: str,
        max_lines: int = 10,
        *,
        success: bool,
    ) -> list[Text]:
        elements: list[Text] = []
        if not content.strip():
            elements.append(Text(" └ (empty)", style="dim"))
            return elements

        lines = content.strip().split("\n")
        display_lines = lines[:max_lines]

        for i, line in enumerate(display_lines):
            prefix = "└" if i == 0 else " "
            rendered = line
            if len(rendered) > 80:
                rendered = rendered[:77] + "..."
            if success:
                elements.append(Text(f" {prefix} {rendered}", style="dim"))
            else:
                elements.append(Text(f" {prefix} {rendered}", style="red dim"))

        remaining = len(lines) - max_lines
        if remaining > 0:
            elements.append(Text(f" ... +{remaining} lines", style="dim italic"))

        return elements

    def _start_tool_call_block(call_id: str, tool_name: str):
        state = tool_call_state.get(call_id)
        if state is not None:
            return
        tool_call_state[call_id] = {
            "input_written": False,
            "result_header_written": False,
        }
        tool_call_meta[call_id] = {
            "name": tool_name,
            "args": {},
            "input_chunks": [],
            "result_written": False,
        }
        renderer.add_ordered_tools_renderable(
            _render_tool_call_line(tool_name, {}, None)
        )

    def _append_tool_input_delta(call_id: str, partial: str):
        if not partial:
            return
        state = tool_call_state.setdefault(
            call_id,
            {"input_written": False, "result_header_written": False},
        )
        state["input_written"] = True
        meta = tool_call_meta.setdefault(
            call_id,
            {
                "name": "unknown",
                "args": {},
                "input_chunks": [],
                "result_written": False,
            },
        )
        chunks = meta.setdefault("input_chunks", [])
        if isinstance(chunks, list):
            chunks.append(partial)

    def _set_tool_input_object(call_id: str, tool_input: Any):
        state = tool_call_state.setdefault(
            call_id,
            {"input_written": False, "result_header_written": False},
        )
        if state["input_written"]:
            return
        meta = tool_call_meta.setdefault(
            call_id,
            {
                "name": "unknown",
                "args": {},
                "input_chunks": [],
                "result_written": False,
            },
        )
        if isinstance(tool_input, dict):
            meta["args"] = tool_input
        else:
            meta["args"] = {}
        state["input_written"] = True

    def _ensure_tool_result_header(call_id: str):
        return

    def _append_tool_result(call_id: str, result_content: Any, *, success: bool = True):
        meta = tool_call_meta.setdefault(
            call_id,
            {
                "name": "unknown",
                "args": {},
                "input_chunks": [],
                "result_written": False,
            },
        )
        if bool(meta.get("result_written")):
            return

        chunks = meta.get("input_chunks")
        if (not meta.get("args")) and isinstance(chunks, list) and chunks:
            joined = "".join(str(x) for x in chunks)
            try:
                parsed = json.loads(joined)
                if isinstance(parsed, dict):
                    meta["args"] = parsed
            except json.JSONDecodeError:
                pass

        if isinstance(result_content, list):
            parts = []
            for item in result_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            result_text = "\n".join(p for p in parts if p)
        elif isinstance(result_content, dict):
            try:
                result_text = json.dumps(result_content, ensure_ascii=False, indent=2)
            except TypeError:
                result_text = str(result_content)
        else:
            result_text = str(result_content or "")

        tool_name = str(meta.get("name") or "unknown")
        tool_args = meta.get("args")
        tr = {"name": tool_name, "content": result_text, "success": success}
        renderer.add_ordered_tools_renderable(
            _render_tool_call_line(tool_name, tool_args, tr)
        )
        is_ok = _is_success(result_text, success)
        if not is_ok:
            for line in _format_tool_result_compact(
                tool_name,
                result_text,
                max_lines=10,
                success=is_ok,
            ):
                renderer.add_ordered_tools_renderable(line)
        meta["result_written"] = True

    def _flush_thinking_buffer():
        if not thinking_buffer:
            return
        text = "".join(thinking_buffer).strip()
        thinking_buffer.clear()
        if not text:
            return
        renderer.add_ordered_thinking(text)

    def _handle_normalized_event(event: StreamEvent):
        if event.type == "tool_call":
            tool_name = str(event.data.get("name") or "unknown")
            tool_args = event.data.get("args") or {}
            tool_call_id = str(event.data.get("id") or "")
            raw_index = event.data.get("index")

            if not tool_call_id:
                tool_call_id = f"anon_{len(tool_call_state) + 1}"

            if isinstance(raw_index, int):
                tool_index_to_id[raw_index] = tool_call_id

            _start_tool_call_block(tool_call_id, tool_name)
            _set_tool_input_object(tool_call_id, tool_args)
            meta = tool_call_meta.get(tool_call_id)
            if meta is not None:
                meta["name"] = tool_name
                if isinstance(tool_args, dict):
                    meta["args"] = tool_args
            return

        if event.type == "tool_result":
            tool_call_id = str(event.data.get("id") or "")
            tool_name = str(event.data.get("name") or "unknown")
            content = event.data.get("content") or ""
            success = bool(event.data.get("success", True))
            if not tool_call_id:
                tool_call_id = f"result_{len(tool_call_state) + 1}"
                _start_tool_call_block(tool_call_id, tool_name)
                _set_tool_input_object(tool_call_id, {})
            elif tool_call_id not in tool_call_state:
                _start_tool_call_block(tool_call_id, tool_name)
                _set_tool_input_object(tool_call_id, {})

            _ensure_tool_result_header(tool_call_id)
            meta = tool_call_meta.get(tool_call_id)
            if meta is not None and not meta.get("name"):
                meta["name"] = tool_name
            _append_tool_result(tool_call_id, content, success=success)

    def _reader_thread(name: str, stream):
        try:
            for line in iter(stream.readline, ""):
                events.put((name, line))
        finally:
            events.put((name, None))

    stdout_thread = threading.Thread(
        target=_reader_thread,
        args=("stdout", proc.stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_reader_thread,
        args=("stderr", proc.stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    stream_closed = {"stdout": False, "stderr": False}
    deadline = start + timeout_sec
    kill_deadline: float | None = None

    with AgentOutputBox(
        StreamRenderConfig(
            title=render_title,
            enabled=stream_output,
            full_output_path=str(live_stream_log),
            transient=transient_output,
        )
    ) as renderer:
        while True:
            if (
                proc.poll() is not None
                and stream_closed["stdout"]
                and stream_closed["stderr"]
            ):
                _flush_thinking_buffer()
                break

            now = time.monotonic()
            if now >= deadline and not timed_out:
                timed_out = True
                kill_deadline = now + 5.0
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except OSError:
                    pass

            if (
                timed_out
                and kill_deadline is not None
                and now >= kill_deadline
                and proc.poll() is None
            ):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                kill_deadline = None

            renderer.update_elapsed(int(now - start))

            try:
                name, payload = events.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                stream_closed[name] = True
                continue

            if name == "stdout":
                stdout_lines.append(payload)
                try:
                    with live_stream_log.open("a", encoding="utf-8") as fp:
                        fp.write(payload)
                except OSError:
                    pass
                text = payload.strip()
                if not text:
                    continue

                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    renderer.add_stdout(payload)
                    continue

                kind = item.get("type")
                event = item.get("event") or {}
                event_type = event.get("type")
                delta = event.get("delta") or {}
                delta_type = delta.get("type")

                # Keep thinking buffered and flush when event type changes.
                if (
                    kind == "stream_event"
                    and event_type == "content_block_delta"
                    and delta_type == "thinking_delta"
                ):
                    thinking_buffer.append(delta.get("thinking") or "")
                    continue

                _flush_thinking_buffer()

                for normalized_event in normalize_raw_messages(item):
                    _handle_normalized_event(normalized_event)

                """解析 stream-json 输出"""
                if kind == "system":
                    status = item.get("status")
                    subtype = item.get("subtype")
                    if status:
                        renderer.set_status(status)
                    elif subtype:
                        renderer.set_status(subtype)
                    continue

                if kind == "stream_event":
                    if event_type == "content_block_start":
                        continue

                    if event_type == "message_start":
                        message = event.get("message") or {}
                        msg_id = message.get("id")
                        current_assistant_stream_id = (
                            str(msg_id) if msg_id is not None else None
                        )
                        continue

                    if event_type == "message_stop":
                        current_assistant_stream_id = None
                        continue

                    if event_type == "content_block_delta":
                        if delta_type == "text_delta":
                            chunk = delta.get("text") or ""
                            if chunk:
                                renderer.add_ordered_assistant(
                                    chunk,
                                    stream_id=current_assistant_stream_id,
                                )
                        elif delta_type == "input_json_delta":
                            partial = delta.get("partial_json") or ""
                            if partial:
                                renderer.set_status("tool input streaming")
                            block_index = event.get("index")
                            if isinstance(block_index, int):
                                tool_call_id = tool_index_to_id.get(block_index)
                                if tool_call_id:
                                    _append_tool_input_delta(tool_call_id, partial)
                        continue

                    if event_type == "content_block_stop":
                        block_index = event.get("index")
                        if isinstance(block_index, int):
                            tool_call_id = tool_index_to_id.get(block_index)
                            if tool_call_id:
                                _ensure_tool_result_header(tool_call_id)
                        continue

                    if event_type == "message_delta":
                        stop_reason = (event.get("delta") or {}).get("stop_reason")
                        renderer.set_usage(event.get("usage"))
                        if stop_reason:
                            renderer.set_status(f"stop_reason={stop_reason}")
                        continue

                    continue

                if kind == "assistant":
                    message = item.get("message") or {}
                    for block in message.get("content") or []:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name") or "tool"
                            renderer.set_status(f"tool_use={tool_name}")
                    continue

                if kind == "user":
                    continue

                if kind == "result":
                    subtype = item.get("subtype")
                    if subtype:
                        renderer.set_status(f"result={subtype}")
                    renderer.set_usage(
                        item.get("usage"),
                        total_cost_usd=item.get("total_cost_usd"),
                    )
                    continue

                renderer.add_stdout(payload)
            else:
                stderr_lines.append(payload)
                try:
                    with live_stream_log.open("a", encoding="utf-8") as fp:
                        fp.write(payload)
                except OSError:
                    pass
                renderer.add_stderr(payload)

    if timed_out and proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)

    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()

    stdout_bytes = "".join(stdout_lines).encode("utf-8", errors="replace")
    stderr_bytes = "".join(stderr_lines).encode("utf-8", errors="replace")

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
        stream_output: bool = True,
    ):
        self._model = model
        self._timeout_sec = timeout_sec
        self._extra_args = extra_args or []
        self._binary = "claude"
        self._stream_output = stream_output

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
        content = _extract_content_from_stream_json(stdout)
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
            "-p",
            prompt,
            "--append-system-prompt",
            CHINESE_THINKING_SYSTEM_PROMPT,
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--allowed-tools",
            "Bash Edit Write Read",
            "--add-dir",
            str(workdir),
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
        render_title: str,
        transient_output: bool = False,
    ) -> tuple[int, str, str, float, bool]:
        return _run_subprocess(
            cmd,
            workdir,
            timeout_sec,
            render_title=render_title,
            stream_output=self._stream_output,
            transient_output=transient_output,
        )

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
5. 你在推进分支以便迭代；如果你感觉在某个方向陷入困境，可以回滚，但必须非常谨慎，并仅在有明确理由时进行。

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
2. 在提出改动前，先读取 autorunner/artifacts/current_state.md（若文件存在）。
3. 使用 Bash 工具定位最新的 autorunner/artifacts/eval-*.json（若存在），并用 Read 工具重点查看 recommendation 与 diagnosis。
4. 若最新评估中的 recommendation.next_action 是 revert_and_retry 或 discard_and_pivot，必须在方案里显式响应该建议。
5. 使用 Edit/Write 工具直接修改 train.py（原地修改，不新建同义副本）。
6. 可用 Bash 工具做轻量自检（例如 python -m py_compile train.py）。
7. 不要只在对话里返回代码块；最终结果应体现在文件系统中的 train.py 里。
8. 无论是否回退到历史版本、重写文件或做最小修补，train.py 顶部都必须保留这一行：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"。若缺失，必须在提交前补回。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述本轮实验改动，中文，5-18词，不含制表符>
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
        summaries_text = (
            "\n".join(run_summaries[-10:]) if run_summaries else "无历史记录"
        )
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
5. 你在推进分支以便迭代；如果你感觉在某个方向陷入困境，可以回滚，但必须非常谨慎，并仅在有明确理由时进行。

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
2. 使用 Read 工具读取 autorunner/artifacts/current_state.md（若文件存在）。
3. 使用 Bash 工具定位最新的 autorunner/artifacts/eval-*.json（若存在），并用 Read 工具重点查看 recommendation 与 diagnosis。
4. 在改动方案中显式说明如何响应最新评估建议（continue/revert_and_retry/discard_and_pivot/investigate）。
5. 使用 Edit/Write 工具直接修改 train.py（原地修改，不新建同义副本）。
6. 可用 Bash 工具做轻量自检（例如 python -m py_compile train.py）。
7. 不要只在对话里返回代码块；最终结果应体现在文件系统中的 train.py 里。
8. 无论是否回退到历史版本、重写文件或做最小修补，train.py 顶部都必须保留这一行：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"。若缺失，必须在提交前补回。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述本轮实验改动，中文，5-18词，不含制表符>
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
5. 无论是否回退到历史版本、重写文件或做最小修补，train.py 顶部都必须保留这一行：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"。若缺失，必须在提交前补回。

## 输出要求
请按以下格式输出，且不要粘贴整份代码：
DESCRIPTION: <一句话描述修复动作，中文，5-18词，不含制表符>
仅使用纯文本，不要使用任何 Markdown 语法符号（例如 #、-、*、```）。
"""
        return prompt

    def _evaluate_prompt(
        self,
        *,
        iteration: int,
        iter_artifact_path: Path,
        run_log_path: Path,
        current_state_path: Path,
        results_tsv_path: Path,
        best_candidate_path: Path,
        eval_output_path: Path,
    ) -> str:
        """生成独立 Evaluator 的 prompt。"""
        return f"""你是 AutoResearch 项目的独立 Evaluator Agent。

你的任务：对刚刚结束的实验进行批判性分析，不要宽容，不要假设作者是正确的。

当前轮次: {iteration}

请按以下顺序执行：
1. 使用 Read 工具读取 {iter_artifact_path}
2. 使用 Read 工具读取 {run_log_path}（完整日志，不要只看尾部）
3. 使用 Bash 工具运行 git diff HEAD~1 train.py，查看本轮代码改动
4. 使用 Read 工具读取 {current_state_path}（如果存在）
5. 使用 Read 工具读取 {results_tsv_path}
6. 使用 Read 工具读取 {best_candidate_path}（如果存在）

分析维度：
A. 实验结果归类：completed_success / completed_anomaly / crashed / timeout / oom
B. 指标对比：本轮 val_bpb 与历史 best 的差异及百分比
C. 训练健康度：从 run.log 诊断是否有 loss spike、梯度异常、I/O 瓶颈
D. 代码审查：本轮 diff 是否合理、是否有 bug、是否过度复杂
E. 下一轮建议：从 continue / revert_and_retry / discard_and_pivot / investigate 中选择，并给出理由

输出要求（必须遵守）：
1. 必须写 JSON 文件到 {eval_output_path}，并严格包含以下字段：
{{
    "iteration": {iteration},
    "evaluated_at": "ISO8601 时间",
    "outcome": "completed_success|completed_anomaly|crashed|timeout|oom",
    "metrics": {{
        "val_bpb": "number|null",
        "best_val_bpb": "number|null",
        "relative_change_percent": "number|null",
        "peak_vram_mb": "number|null",
        "mfu_percent": "number|null"
    }},
    "metrics_healthy": "bool",
    "diagnosis": {{
        "summary": "string",
        "issues": ["string", "..."],
        "root_cause": "string"
    }},
    "code_review": {{
        "diff_summary": "string",
        "risks": ["string", "..."],
        "consistency": "bool"
    }},
    "recommendation": {{
        "next_action": "continue|revert_and_retry|discard_and_pivot|investigate",
        "reasoning": "string",
        "suggested_directions": ["string", "..."]
    }}
}}
2. 必须更新 {current_state_path}，追加本轮结论、主要问题和下一轮推荐方向。
3. 不要修改 train.py。
4. 终端输出只需一句话总结本轮结论。
"""

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
            topic,
            exp_plan,
            metric_key,
            pkg_hint,
            compute_budget,
            extra_guidance,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd,
            workdir,
            timeout_sec or self._timeout_sec,
            render_title="Generator (generate)",
            transient_output=True,
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
            current_files,
            run_summaries,
            metric_key,
            metric_direction,
            topic,
            extra_hints,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd,
            workdir,
            timeout_sec or self._timeout_sec,
            render_title="Generator (refine)",
            transient_output=True,
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
            cmd,
            workdir,
            timeout_sec or self._timeout_sec,
            render_title="Generator (repair)",
            transient_output=True,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def evaluate(
        self,
        *,
        iteration: int,
        iter_artifact_path: Path,
        run_log_path: Path,
        current_state_path: Path,
        results_tsv_path: Path,
        best_candidate_path: Path,
        eval_output_path: Path,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        prompt = self._evaluate_prompt(
            iteration=iteration,
            iter_artifact_path=iter_artifact_path,
            run_log_path=run_log_path,
            current_state_path=current_state_path,
            results_tsv_path=results_tsv_path,
            best_candidate_path=best_candidate_path,
            eval_output_path=eval_output_path,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd,
            workdir,
            timeout_sec or self._timeout_sec,
            render_title="Evaluator",
            transient_output=True,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)
