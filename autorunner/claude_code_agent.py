"""ClaudeCodeAgent: 调用 Claude Code CLI (claude -p) 生成/修改 train.py"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.text import Text

from autorunner.terminal_box import AgentOutputBox, StreamRenderConfig


DEFAULT_PROMPT_CONFIG_NAME = "subagent.yaml"


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
        """构造工具调用事件对象。"""
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
        """构造工具结果事件对象。"""
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
    """将原始流式消息标准化为统一事件列表。"""
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
    """将字节数据安全解码为文本。"""
    if data is None:
        return ""
    return data.decode("utf-8", errors="replace")


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """读取工作目录下所有顶层 Python 文件内容。"""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    return files


def _extract_content_from_stream_json(stdout: str) -> str:
    """从 stream-json 输出中提取最终回复文本。"""
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


def _extract_error_from_stream_json(stdout: str) -> str:
    """从 stream-json 输出中提取可读的错误信息。"""
    error_text = ""

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = obj.get("type")
        if kind == "result":
            if bool(obj.get("is_error")):
                result_text = str(obj.get("result") or "").strip()
                if result_text:
                    error_text = result_text
            continue

        if kind == "assistant" and obj.get("error"):
            message = obj.get("message") or {}
            blocks = message.get("content") or []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = str(block.get("text") or "").strip()
                if text:
                    error_text = text

    return error_text


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


@dataclass
class _SubprocessStreamState:
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    events: queue.Queue[tuple[str, str | None]] = field(default_factory=queue.Queue)
    tool_index_to_id: dict[int, str] = field(default_factory=dict)
    tool_call_state: dict[str, dict[str, bool]] = field(default_factory=dict)
    tool_call_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    thinking_buffer: list[str] = field(default_factory=list)
    current_assistant_stream_id: str | None = None
    stream_closed: dict[str, bool] = field(
        default_factory=lambda: {"stdout": False, "stderr": False}
    )
    timed_out: bool = False
    kill_deadline: float | None = None


def _append_live_stream_line(log_path: Path, payload: str) -> None:
    """将已解析的实时流事件追加写入 Markdown 日志文件。"""
    try:
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(payload.rstrip("\n") + "\n")
    except OSError:
        pass


def _short_text(text: Any, max_len: int = 200) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _append_live_stream_event(log_path: Path, kind: str, **fields: Any) -> None:
    """按统一结构写入实时流 Markdown 事件。"""
    if kind == "assistant_text":
        _append_live_stream_line(
            log_path, f"- assistant: {_short_text(fields.get('text'))}"
        )
        return
    if kind == "thinking":
        _append_live_stream_line(
            log_path, f"- thinking: {_short_text(fields.get('text'))}"
        )
        return
    if kind == "tool_call":
        name = _short_text(fields.get("name"), 80)
        call_id = _short_text(fields.get("id"), 80)
        args = fields.get("args") or {}
        try:
            args_text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            args_text = str(args)
        _append_live_stream_line(
            log_path,
            f"- tool_call: name={name} id={call_id} args={_short_text(args_text, 800)}",
        )
        return
    if kind == "tool_result":
        name = _short_text(fields.get("name"), 80)
        ok = bool(fields.get("success", True))
        call_id = _short_text(fields.get("id"), 80)
        content = fields.get("content") or ""
        _append_live_stream_line(
            log_path,
            f"- tool_result: name={name} id={call_id} success={ok}",
        )
        _append_live_stream_line(log_path, "")
        _append_live_stream_line(log_path, "```text")
        _append_live_stream_line(log_path, str(content))
        _append_live_stream_line(log_path, "```")
        return
    if kind == "stderr":
        _append_live_stream_line(
            log_path, f"- stderr: {_short_text(fields.get('text'), 1000)}"
        )
        return
    if kind == "stdout":
        _append_live_stream_line(
            log_path, f"- stdout: {_short_text(fields.get('text'), 1000)}"
        )
        return
    if kind == "system_status":
        _append_live_stream_line(
            log_path, f"- system_status: {_short_text(fields.get('status'))}"
        )
        return
    if kind == "system_subtype":
        _append_live_stream_line(
            log_path, f"- system_subtype: {_short_text(fields.get('subtype'))}"
        )
        return
    if kind == "result":
        subtype = _short_text(fields.get("subtype"), 80)
        is_error = bool(fields.get("is_error", False))
        _append_live_stream_line(
            log_path, f"- result: subtype={subtype} is_error={is_error}"
        )
        return
    _append_live_stream_line(log_path, f"- {kind}: {_short_text(fields)}")


def _stream_reader_thread(
    name: str,
    stream: Any,
    events: queue.Queue[tuple[str, str | None]],
) -> None:
    """持续读取子进程输出流并投递到事件队列。"""
    try:
        for line in iter(stream.readline, ""):
            events.put((name, line))
    finally:
        events.put((name, None))


def _start_stream_reader_threads(
    proc: subprocess.Popen[str],
    events: queue.Queue[tuple[str, str | None]],
) -> None:
    """启动标准输出与标准错误的后台读取线程。"""
    stdout_thread = threading.Thread(
        target=_stream_reader_thread,
        args=("stdout", proc.stdout, events),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_reader_thread,
        args=("stderr", proc.stderr, events),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()


def _terminate_subprocess_group(proc: subprocess.Popen[str], timed_out: bool) -> None:
    """按超时状态优雅终止并回收子进程相关资源。"""
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


class _StreamOutputProcessor:
    def __init__(self, state: _SubprocessStreamState, renderer: Any):
        """初始化流输出处理器。"""
        self.state = state
        self.renderer = renderer

    @staticmethod
    def _format_tool_compact(name: str, args: dict[str, Any] | None) -> str:
        """将工具名和参数格式化为紧凑展示文本。"""
        if not args:
            return name
        try:
            compact = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            compact = str(args)
        if len(compact) > 120:
            compact = compact[:117] + "..."
        return f"{name} {compact}"

    @staticmethod
    def _is_success(content: str, success_flag: bool) -> bool:
        """根据标记与内容判断工具调用是否成功。"""
        if not success_flag:
            return False
        lowered = content.lower()
        return (
            "<tool_use_error>" not in lowered and "inputvalidationerror" not in lowered
        )

    def _render_tool_call_line(
        self,
        name: str,
        args: dict[str, Any],
        tr: dict[str, Any] | None,
    ) -> Text:
        """渲染单条工具调用状态行。"""
        is_task = name.lower() == "task"
        if tr is not None:
            content = str(tr.get("content") or "")
            if self._is_success(content, bool(tr.get("success", True))):
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
        tool_text.append(self._format_tool_compact(name, args), style=style)
        return tool_text

    def _format_tool_result_compact(
        self,
        content: str,
        max_lines: int,
        *,
        success: bool,
    ) -> list[Text]:
        """将工具结果压缩为适合终端展示的文本行。"""
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

    def _start_tool_call_block(self, call_id: str, tool_name: str) -> None:
        """初始化并渲染一个工具调用块。"""
        if call_id in self.state.tool_call_state:
            return
        self.state.tool_call_state[call_id] = {
            "input_written": False,
            "result_header_written": False,
        }
        self.state.tool_call_meta[call_id] = {
            "name": tool_name,
            "args": {},
            "input_chunks": [],
            "result_written": False,
        }
        self.renderer.add_ordered_tools_renderable(
            self._render_tool_call_line(tool_name, {}, None)
        )

    def _append_tool_input_delta(self, call_id: str, partial: str) -> None:
        """追加工具输入的增量 JSON 片段。"""
        if not partial:
            return
        state = self.state.tool_call_state.setdefault(
            call_id,
            {"input_written": False, "result_header_written": False},
        )
        state["input_written"] = True
        meta = self.state.tool_call_meta.setdefault(
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

    def _set_tool_input_object(self, call_id: str, tool_input: Any) -> None:
        """记录工具调用的完整输入对象。"""
        state = self.state.tool_call_state.setdefault(
            call_id,
            {"input_written": False, "result_header_written": False},
        )
        if state["input_written"]:
            return
        meta = self.state.tool_call_meta.setdefault(
            call_id,
            {
                "name": "unknown",
                "args": {},
                "input_chunks": [],
                "result_written": False,
            },
        )
        meta["args"] = tool_input if isinstance(tool_input, dict) else {}
        state["input_written"] = True

    def _ensure_tool_result_header(self, call_id: str) -> None:
        """预留工具结果头部处理钩子。"""
        _ = call_id
        return

    def _append_tool_result(
        self,
        call_id: str,
        result_content: Any,
        *,
        success: bool = True,
    ) -> None:
        """写入并渲染工具调用结果。"""
        meta = self.state.tool_call_meta.setdefault(
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
        self.renderer.add_ordered_tools_renderable(
            self._render_tool_call_line(tool_name, tool_args, tr)
        )
        is_ok = self._is_success(result_text, success)
        if not is_ok:
            for line in self._format_tool_result_compact(
                result_text,
                max_lines=10,
                success=is_ok,
            ):
                self.renderer.add_ordered_tools_renderable(line)
        meta["result_written"] = True

    def flush_thinking_buffer(self) -> None:
        """刷新并渲染暂存的思考文本。"""
        if not self.state.thinking_buffer:
            return
        text = "".join(self.state.thinking_buffer).strip()
        self.state.thinking_buffer.clear()
        if text:
            self.renderer.add_ordered_thinking(text)

    def _handle_normalized_event(self, event: StreamEvent) -> None:
        """处理标准化后的工具事件。"""
        if event.type == "tool_call":
            tool_name = str(event.data.get("name") or "unknown")
            tool_args = event.data.get("args") or {}
            tool_call_id = str(event.data.get("id") or "")
            raw_index = event.data.get("index")

            if not tool_call_id:
                tool_call_id = f"anon_{len(self.state.tool_call_state) + 1}"

            if isinstance(raw_index, int):
                self.state.tool_index_to_id[raw_index] = tool_call_id

            self._start_tool_call_block(tool_call_id, tool_name)
            self._set_tool_input_object(tool_call_id, tool_args)
            meta = self.state.tool_call_meta.get(tool_call_id)
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
                tool_call_id = f"result_{len(self.state.tool_call_state) + 1}"
                self._start_tool_call_block(tool_call_id, tool_name)
                self._set_tool_input_object(tool_call_id, {})
            elif tool_call_id not in self.state.tool_call_state:
                self._start_tool_call_block(tool_call_id, tool_name)
                self._set_tool_input_object(tool_call_id, {})

            self._ensure_tool_result_header(tool_call_id)
            meta = self.state.tool_call_meta.get(tool_call_id)
            if meta is not None and not meta.get("name"):
                meta["name"] = tool_name
            self._append_tool_result(tool_call_id, content, success=success)

    def _log_normalized_event(self, event: StreamEvent, live_stream_log: Path) -> None:
        """将标准化事件写入 live stream 日志。"""
        if event.type == "tool_call":
            _append_live_stream_event(
                live_stream_log,
                "tool_call",
                name=str(event.data.get("name") or "unknown"),
                id=str(event.data.get("id") or ""),
                args=(event.data.get("args") or {}),
            )
            return

        if event.type == "tool_result":
            content = str(event.data.get("content") or "")
            _append_live_stream_event(
                live_stream_log,
                "tool_result",
                name=str(event.data.get("name") or "unknown"),
                id=str(event.data.get("id") or ""),
                success=bool(event.data.get("success", True)),
                content=content,
            )

    def handle_stdout_payload(self, payload: str, live_stream_log: Path) -> None:
        """处理并渲染标准输出中的单条负载。"""
        self.state.stdout_lines.append(payload)

        text = payload.strip()
        if not text:
            return

        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            _append_live_stream_event(live_stream_log, "stdout", text=text)
            self.renderer.add_stdout(payload)
            return

        kind = item.get("type")
        event = item.get("event") or {}
        event_type = event.get("type")
        delta = event.get("delta") or {}
        delta_type = delta.get("type")

        if (
            kind == "stream_event"
            and event_type == "content_block_delta"
            and delta_type == "thinking_delta"
        ):
            thinking = delta.get("thinking") or ""
            self.state.thinking_buffer.append(thinking)
            if thinking:
                _append_live_stream_event(live_stream_log, "thinking", text=thinking)
            return

        self.flush_thinking_buffer()

        for normalized_event in normalize_raw_messages(item):
            self._handle_normalized_event(normalized_event)
            self._log_normalized_event(normalized_event, live_stream_log)

        if kind == "system":
            status = item.get("status")
            subtype = item.get("subtype")
            if status:
                _append_live_stream_event(
                    live_stream_log,
                    "system_status",
                    status=str(status),
                )
                self.renderer.set_status(status)
            elif subtype:
                _append_live_stream_event(
                    live_stream_log,
                    "system_subtype",
                    subtype=str(subtype),
                )
                self.renderer.set_status(subtype)
            return

        if kind == "stream_event":
            if event_type == "content_block_start":
                return

            if event_type == "message_start":
                message = event.get("message") or {}
                msg_id = message.get("id")
                self.state.current_assistant_stream_id = (
                    str(msg_id) if msg_id is not None else None
                )
                return

            if event_type == "message_stop":
                self.state.current_assistant_stream_id = None
                return

            if event_type == "content_block_delta":
                if delta_type == "text_delta":
                    chunk = delta.get("text") or ""
                    if chunk:
                        _append_live_stream_event(
                            live_stream_log,
                            "assistant_text",
                            text=chunk,
                        )
                        self.renderer.add_ordered_assistant(
                            chunk,
                            stream_id=self.state.current_assistant_stream_id,
                        )
                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json") or ""
                    if partial:
                        self.renderer.set_status("tool input streaming")
                    block_index = event.get("index")
                    if isinstance(block_index, int):
                        tool_call_id = self.state.tool_index_to_id.get(block_index)
                        if tool_call_id:
                            self._append_tool_input_delta(tool_call_id, partial)
                return

            if event_type == "content_block_stop":
                block_index = event.get("index")
                if isinstance(block_index, int):
                    tool_call_id = self.state.tool_index_to_id.get(block_index)
                    if tool_call_id:
                        self._ensure_tool_result_header(tool_call_id)
                return

            if event_type == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason")
                self.renderer.set_usage(event.get("usage"))
                if stop_reason:
                    self.renderer.set_status(f"stop_reason={stop_reason}")
                return

            return

        if kind == "assistant":
            message = item.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_name = block.get("name") or "tool"
                    self.renderer.set_status(f"tool_use={tool_name}")
            return

        if kind == "user":
            return

        if kind == "result":
            subtype = item.get("subtype")
            if subtype:
                _append_live_stream_event(
                    live_stream_log,
                    "result",
                    subtype=str(subtype),
                    is_error=bool(item.get("is_error", False)),
                )
                self.renderer.set_status(f"result={subtype}")
            self.renderer.set_usage(
                item.get("usage"),
                total_cost_usd=item.get("total_cost_usd"),
            )
            return

        _append_live_stream_event(live_stream_log, "stdout", text=text)
        self.renderer.add_stdout(payload)

    def handle_stderr_payload(self, payload: str, live_stream_log: Path) -> None:
        """处理并渲染标准错误中的单条负载。"""
        self.state.stderr_lines.append(payload)
        text = payload.strip()
        if text:
            _append_live_stream_event(live_stream_log, "stderr", text=text)
        self.renderer.add_stderr(payload)


def _run_subprocess(
    cmd: list[str],
    workdir: Path,
    timeout_sec: int,
    render_title: str = "Claude Agent Output",
    stream_output: bool = True,
    transient_output: bool = False,
) -> tuple[int, str, str, float, bool]:
    """运行子进程并返回退出码、输出与超时信息。"""
    workdir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    artifacts_dir = workdir / "autorunner" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (
        render_title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    live_stream_log = artifacts_dir / f"live_stream_{safe_title}_{int(start)}.md"
    _append_live_stream_line(live_stream_log, f"# {render_title}")
    _append_live_stream_line(live_stream_log, f"- started_at: {int(start)}")
    _append_live_stream_line(live_stream_log, f"- workdir: {workdir}")
    _append_live_stream_line(live_stream_log, "")

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

    state = _SubprocessStreamState()
    _start_stream_reader_threads(proc, state.events)

    deadline = start + timeout_sec

    with AgentOutputBox(
        StreamRenderConfig(
            title=render_title,
            enabled=stream_output,
            full_output_path=str(live_stream_log),
            transient=transient_output,
        )
    ) as renderer:
        processor = _StreamOutputProcessor(state, renderer)
        while True:
            if (
                proc.poll() is not None
                and state.stream_closed["stdout"]
                and state.stream_closed["stderr"]
            ):
                processor.flush_thinking_buffer()
                break

            now = time.monotonic()
            if now >= deadline and not state.timed_out:
                state.timed_out = True
                state.kill_deadline = now + 5.0
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except OSError:
                    pass

            if (
                state.timed_out
                and state.kill_deadline is not None
                and now >= state.kill_deadline
                and proc.poll() is None
            ):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                state.kill_deadline = None

            renderer.update_elapsed(int(now - start))

            try:
                name, payload = state.events.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                state.stream_closed[name] = True
                continue

            if name == "stdout":
                processor.handle_stdout_payload(payload, live_stream_log)
            else:
                processor.handle_stderr_payload(payload, live_stream_log)

    _terminate_subprocess_group(proc, state.timed_out)

    stdout_bytes = "".join(state.stdout_lines).encode("utf-8", errors="replace")
    stderr_bytes = "".join(state.stderr_lines).encode("utf-8", errors="replace")

    elapsed = time.monotonic() - start
    return (
        proc.returncode if proc.returncode is not None else -1,
        _to_text(stdout_bytes),
        _to_text(stderr_bytes),
        elapsed,
        state.timed_out,
    )


class ClaudeCodeAgent:
    """Backed by Claude Code CLI (claude -p)."""

    def __init__(
        self,
        model: str = "sonnet",
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        stream_output: bool = True,
        prompt_config_path: Path | None = None,
    ):
        """初始化 Claude Code Agent 的运行参数与提示词配置。"""
        self._model = model
        self._timeout_sec = timeout_sec
        self._extra_args = extra_args or []
        self._binary = "claude"
        self._stream_output = stream_output
        self._prompt_config_path = prompt_config_path or Path(__file__).with_name(
            DEFAULT_PROMPT_CONFIG_NAME
        )
        self._prompt_templates = self._load_prompt_templates(self._prompt_config_path)

    def _load_prompt_templates(self, config_path: Path) -> dict[str, Any]:
        """加载并校验提示词模板配置。"""
        if not config_path.exists():
            raise FileNotFoundError(f"Prompt config file not found: {config_path}")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Prompt config must be a mapping: {config_path}")
        return data

    def _render_prompt(self, section: str, template_name: str, **kwargs: Any) -> str:
        """按模板与变量渲染最终提示词。"""
        section_map = self._prompt_templates.get(section)
        if not isinstance(section_map, dict):
            raise ValueError(f"Missing prompt section: {section}")

        template = section_map.get(template_name)
        if not isinstance(template, str):
            raise ValueError(f"Missing prompt template: {section}.{template_name}")

        try:
            return template.format(**kwargs)
        except KeyError as exc:
            missing_key = exc.args[0]
            raise ValueError(
                f"Missing prompt variable '{missing_key}' for {section}.{template_name}"
            ) from exc

    def _build_result(
        self,
        workdir: Path,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        timed_out: bool,
    ) -> CodeAgentResult:
        """汇总执行产物并构造统一结果对象。"""
        files = _collect_py_files(workdir)
        effective_stderr = stderr.strip()
        if returncode != 0 and not effective_stderr:
            effective_stderr = _extract_error_from_stream_json(stdout)

        error = None
        if timed_out:
            error = f"Timed out after {elapsed:.0f}s"
        elif returncode != 0:
            error = f"Exited {returncode}: {effective_stderr[:500]}"

        # 日志内容记录 Claude 文本回复，代码改动通过 files['train.py'] 读取
        content = _extract_content_from_stream_json(stdout)
        has_train_code = bool(files.get("train.py", "").strip())
        has_reply = bool(content)

        return CodeAgentResult(
            success=(error is None and (has_train_code or has_reply)),
            content=content,
            rc=returncode,
            stderr=effective_stderr,
            elapsed=elapsed,
            timed_out=timed_out,
            files=files,
        )

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        """组装 Claude CLI 命令行参数。"""
        cmd = [
            self._binary,
            "-p",
            prompt,
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
        """调用底层子进程执行器并透传结果。"""
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
        extra_guidance: str,
    ) -> str:
        """生成首次实验迭代的提示词。"""
        return self._render_prompt(
            "generator",
            "generate",
            topic=topic,
            exp_plan=exp_plan,
            metric_key=metric_key,
            extra_guidance=extra_guidance,
        )

    def _refine_prompt(
        self,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
    ) -> str:
        """基于历史结果生成 refine 提示词。"""
        summaries_text = (
            "\n".join(run_summaries[-10:]) if run_summaries else "无历史记录"
        )
        files_text = ""
        for name, content in current_files.items():
            files_text += f"\n=== {name} ===\n{content[:3000]}"
        return self._render_prompt(
            "generator",
            "refine",
            topic=topic,
            metric_key=metric_key,
            metric_direction=metric_direction,
            summaries_text=summaries_text,
            files_text=files_text,
            extra_hints=extra_hints,
        )

    def _repair_prompt(
        self,
        files: dict[str, str],
        issues: str,
        refine_description: str,
    ) -> str:
        """生成修复问题的提示词。"""
        files_text = ""
        for name, content in files.items():
            files_text += f"\n=== {name} ===\n{content[:3000]}"
        return self._render_prompt(
            "generator",
            "repair",
            files_text=files_text,
            issues=issues,
            refine_description=refine_description,
        )

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
        """生成独立评估代理使用的提示词。"""
        return self._render_prompt(
            "evaluator",
            "evaluate",
            iteration=iteration,
            iter_artifact_path=iter_artifact_path,
            run_log_path=run_log_path,
            current_state_path=current_state_path,
            results_tsv_path=results_tsv_path,
            best_candidate_path=best_candidate_path,
            eval_output_path=eval_output_path,
        )

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        """执行首次生成流程并返回结果。"""
        prompt = self._generate_prompt(
            topic,
            exp_plan,
            metric_key,
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
        """执行迭代优化流程并返回结果。"""
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
        refine_description: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        """执行修复流程并返回结果。"""
        prompt = self._repair_prompt(files, issues, refine_description)
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
        """执行评估流程并返回结果。"""
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
