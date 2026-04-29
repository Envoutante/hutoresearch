"""Code agents backed by Claude Code or Codex CLI."""

from __future__ import annotations

import json
import os
import queue
import signal
import shlex
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.text import Text

from autorunner.terminal_box import AgentOutputBox, StreamRenderConfig


DEFAULT_PROMPT_CONFIG_NAME = "subagent.yaml"
REFINE_SUMMARY_LIMIT = int(os.getenv("AR_REFINE_SUMMARY_LIMIT", "6"))


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


def _extract_content_from_codex_json(stdout: str) -> str:
    """从 Codex CLI --json 输出中提取最终回复文本。"""
    final_text = ""
    text_parts: list[str] = []

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            text_parts.append(line)
            continue

        item = obj.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            text = item.get("text")
            if (
                obj.get("type") == "item.completed"
                and item_type in {"agent_message", "reasoning"}
                and isinstance(text, str)
                and text.strip()
            ):
                if item_type == "agent_message":
                    final_text = text.strip()
                else:
                    text_parts.append(text)

        for key in ("last_message", "message", "content", "text", "output"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                if obj.get("type") in {"result", "final", "agent_message"}:
                    final_text = value.strip()
                else:
                    text_parts.append(value)

        if obj.get("type") == "result":
            result = obj.get("result")
            if isinstance(result, str) and result.strip():
                final_text = result.strip()

    if final_text:
        return final_text
    return "\n".join(x.strip() for x in text_parts if x.strip()).strip()


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
    """代码代理的返回结果。"""

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
    thinking_log_buffer: str = ""
    assistant_log_buffer: str = ""
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
    if max_len <= 0 or len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _append_live_stream_event(log_path: Path, kind: str, **fields: Any) -> None:
    """按统一结构写入实时流 Markdown 事件。"""
    if kind == "assistant_text":
        text = str(fields.get("text") or "")
        if text:
            _append_live_stream_line(log_path, text.rstrip("\n"))
        return
    if kind == "thinking":
        text = str(fields.get("text") or "")
        if text:
            _append_live_stream_line(log_path, text.rstrip("\n"))
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

    @staticmethod
    def _split_complete_units(buffer: str) -> tuple[list[str], str]:
        """从缓冲区中拆出完整句子或行，保留未闭合尾部。"""
        done: list[str] = []
        start = 0
        n = len(buffer)
        i = 0

        while i < n:
            ch = buffer[i]
            if ch == "\n":
                segment = buffer[start:i]
                if segment.strip():
                    done.append(segment.rstrip())
                start = i + 1
                i += 1
                continue

            if ch in ".!?。！？":
                next_ch = buffer[i + 1] if i + 1 < n else ""
                if i + 1 == n or next_ch.isspace() or next_ch in "\"'”’)]}":
                    segment = buffer[start : i + 1]
                    stripped = segment.strip()
                    if ch == "." and stripped[:-1].isdigit() and stripped.endswith("."):
                        i += 1
                        continue
                    if stripped:
                        done.append(segment.rstrip())
                    start = i + 1
            i += 1

        rest = buffer[start:]
        return done, rest

    def _append_stream_text_for_log(
        self,
        kind: str,
        chunk: str,
        live_stream_log: Path,
        *,
        force: bool = False,
    ) -> None:
        """将流式文本聚合为完整句再写入 Markdown。"""
        if kind == "thinking":
            self.state.thinking_log_buffer += chunk
            completed, rest = self._split_complete_units(self.state.thinking_log_buffer)
            self.state.thinking_log_buffer = rest
            for line in completed:
                _append_live_stream_event(live_stream_log, "thinking", text=line)
            if force and self.state.thinking_log_buffer.strip():
                _append_live_stream_event(
                    live_stream_log,
                    "thinking",
                    text=self.state.thinking_log_buffer.rstrip(),
                )
                self.state.thinking_log_buffer = ""
            return

        if kind == "assistant":
            self.state.assistant_log_buffer += chunk
            completed, rest = self._split_complete_units(
                self.state.assistant_log_buffer
            )
            self.state.assistant_log_buffer = rest
            for line in completed:
                _append_live_stream_event(live_stream_log, "assistant_text", text=line)
            if force and self.state.assistant_log_buffer.strip():
                _append_live_stream_event(
                    live_stream_log,
                    "assistant_text",
                    text=self.state.assistant_log_buffer.rstrip(),
                )
                self.state.assistant_log_buffer = ""

    def flush_live_log_buffers(self, live_stream_log: Path) -> None:
        """强制刷新日志缓冲，避免进程结束时残留半句。"""
        self._append_stream_text_for_log(
            "thinking",
            "",
            live_stream_log,
            force=True,
        )
        self._append_stream_text_for_log(
            "assistant",
            "",
            live_stream_log,
            force=True,
        )

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
            tool_call_id = str(event.data.get("id") or "")
            tool_name = str(event.data.get("name") or "unknown")
            tool_args: dict[str, Any] = {}
            if tool_call_id:
                meta = self.state.tool_call_meta.get(tool_call_id) or {}
                meta_name = str(meta.get("name") or "").strip()
                if tool_name in {"", "unknown"} and meta_name:
                    tool_name = meta_name
                raw_args = meta.get("args")
                if isinstance(raw_args, dict):
                    tool_args = raw_args

            if tool_name.lower() == "read":
                file_path = str(
                    tool_args.get("file_path") or tool_args.get("path") or ""
                ).strip()
                line = (
                    f"- tool_result: name=Read id={_short_text(tool_call_id, 80)} "
                    f"success={bool(event.data.get('success', True))}"
                )
                if file_path:
                    line += f" file_path={file_path}"
                _append_live_stream_line(live_stream_log, line)
                return

            content = str(event.data.get("content") or "")
            _append_live_stream_event(
                live_stream_log,
                "tool_result",
                name=tool_name,
                id=tool_call_id,
                success=bool(event.data.get("success", True)),
                content=content,
            )

    def _handle_codex_json_event(self, item: dict[str, Any], live_stream_log: Path) -> bool:
        """记录 Codex CLI --json 事件，省略工具读取到的原始正文。"""
        kind = str(item.get("type") or "")

        if kind == "thread.started":
            thread_id = _short_text(item.get("thread_id"), 80)
            suffix = f" id={thread_id}" if thread_id else ""
            _append_live_stream_line(live_stream_log, f"- thread: started{suffix}")
            self.renderer.set_status("thread started")
            return True

        if kind == "turn.started":
            _append_live_stream_line(live_stream_log, "- turn: started")
            self.renderer.set_status("turn started")
            return True

        if kind == "turn.completed":
            usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
            usage_text = ""
            if usage:
                usage_text = " usage=" + _short_text(
                    json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                    500,
                )
            _append_live_stream_line(live_stream_log, f"- turn: completed{usage_text}")
            self.renderer.set_usage(usage)
            self.renderer.set_status("turn completed")
            return True

        if kind == "error":
            message = _short_text(item.get("message"), 1000)
            _append_live_stream_line(live_stream_log, f"- error: {message}")
            self.renderer.add_stderr(message)
            return True

        if kind not in {"item.started", "item.completed"}:
            return False

        raw_item = item.get("item")
        if not isinstance(raw_item, dict):
            return False

        event_phase = kind.split(".", 1)[1]
        item_id = _short_text(raw_item.get("id"), 80)
        item_type = str(raw_item.get("type") or "unknown")
        status = str(raw_item.get("status") or event_phase)

        if item_type in {"agent_message", "reasoning"}:
            text = str(raw_item.get("text") or "")
            if text:
                log_kind = "thinking" if item_type == "reasoning" else "assistant"
                self._append_stream_text_for_log(
                    log_kind,
                    text,
                    live_stream_log,
                    force=True,
                )
                if item_type == "reasoning":
                    self.renderer.add_ordered_thinking(text)
                else:
                    self.renderer.add_ordered_assistant(text, stream_id=item_id or None)
            return True

        if item_type == "command_execution":
            command = _short_text(raw_item.get("command"), 500)
            if event_phase == "started":
                _append_live_stream_line(
                    live_stream_log,
                    f"- tool_call: name=command_execution id={item_id} command={command}",
                )
                self.renderer.set_status("command running")
                return True

            output = str(raw_item.get("aggregated_output") or "")
            exit_code = raw_item.get("exit_code")
            exit_text = "none" if exit_code is None else str(exit_code)
            _append_live_stream_line(
                live_stream_log,
                (
                    "- tool_result: name=command_execution "
                    f"id={item_id} status={status} exit_code={exit_text} "
                    f"output_chars={len(output)}"
                ),
            )
            self.renderer.set_status(f"command {status}")
            return True

        if item_type == "file_change":
            changes = raw_item.get("changes")
            change_summaries: list[str] = []
            if isinstance(changes, list):
                for change in changes[:8]:
                    if not isinstance(change, dict):
                        continue
                    path = _short_text(change.get("path"), 160)
                    change_kind = _short_text(change.get("kind"), 40)
                    if path and change_kind:
                        change_summaries.append(f"{change_kind}:{path}")
                    elif path:
                        change_summaries.append(path)
                if len(changes) > len(change_summaries):
                    change_summaries.append(f"+{len(changes) - len(change_summaries)} more")
            changes_text = ", ".join(change_summaries) if change_summaries else "-"
            if event_phase == "started":
                _append_live_stream_line(
                    live_stream_log,
                    f"- tool_call: name=file_change id={item_id} changes={changes_text}",
                )
            else:
                _append_live_stream_line(
                    live_stream_log,
                    (
                        "- tool_result: name=file_change "
                        f"id={item_id} status={status} changes={changes_text}"
                    ),
                )
            self.renderer.set_status(f"file_change {status}")
            return True

        _append_live_stream_line(
            live_stream_log,
            f"- codex_item: event={event_phase} type={item_type} id={item_id} status={status}",
        )
        self.renderer.set_status(f"{item_type} {status}")
        return True

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

        if self._handle_codex_json_event(item, live_stream_log):
            return

        if (
            kind == "stream_event"
            and event_type == "content_block_delta"
            and delta_type == "thinking_delta"
        ):
            thinking = delta.get("thinking") or ""
            self.state.thinking_buffer.append(thinking)
            if thinking:
                self._append_stream_text_for_log(
                    "thinking",
                    thinking,
                    live_stream_log,
                )
            return

        self.flush_thinking_buffer()
        self._append_stream_text_for_log("thinking", "", live_stream_log, force=True)

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
                self._append_stream_text_for_log(
                    "assistant",
                    "",
                    live_stream_log,
                    force=True,
                )
                self.state.current_assistant_stream_id = None
                return

            if event_type == "content_block_delta":
                if delta_type == "text_delta":
                    chunk = delta.get("text") or ""
                    if chunk:
                        self._append_stream_text_for_log(
                            "assistant",
                            chunk,
                            live_stream_log,
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
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (
        render_title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    for prefix in ("claude_agent_output_", "generator_", "evaluator_"):
        if safe_title.startswith(prefix):
            safe_title = safe_title[len(prefix) :]
            break
    safe_title = safe_title.strip("_") or "session"
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    live_stream_log = logs_dir / f"{safe_title}-{stamp}.md"
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
                processor.flush_live_log_buffers(live_stream_log)
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


class BaseCodeAgent:
    """Shared prompt rendering and task methods for CLI-backed coding agents."""

    backend_name = "base"
    default_binary = ""
    supports_allowed_tools = False

    def __init__(
        self,
        model: str = "",
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        stream_output: bool = True,
        prompt_config_path: Path | None = None,
        binary: str | None = None,
    ):
        """初始化 Code Agent 的运行参数与提示词配置。"""
        self._model = model
        self._timeout_sec = timeout_sec
        self._extra_args = extra_args or []
        self._binary = binary or self.default_binary
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

    def _extract_content(self, stdout: str) -> str:
        return stdout.strip()

    def _extract_error(self, stdout: str, stderr: str) -> str:
        _ = stdout
        return stderr.strip()

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
        effective_stderr = self._extract_error(stdout, stderr)
        if returncode != 0 and not effective_stderr:
            effective_stderr = stdout.strip()[:500]

        error = None
        if timed_out:
            error = f"Timed out after {elapsed:.0f}s"
        elif returncode != 0:
            error = f"Exited {returncode}: {effective_stderr[:500]}"

        # 日志内容记录 Agent 文本回复，代码改动通过 files['train.py'] 读取
        content = self._extract_content(stdout)
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

    def _build_cmd(
        self,
        prompt: str,
        workdir: Path,
        *,
        allowed_tools: str = "Bash Edit Write Read",
    ) -> list[str]:
        _ = prompt, workdir, allowed_tools
        raise NotImplementedError

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
            "\n".join(run_summaries[-REFINE_SUMMARY_LIMIT:])
            if run_summaries
            else "无历史记录"
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

    def _candidate_summary_prompt(
        self,
        *,
        candidate_id: str,
        diff_summary: str,
    ) -> str:
        """生成候选结构化摘要提示词，用于抢救超时但已修改的候选。"""
        return self._render_prompt(
            "candidate_summarizer",
            "summarize",
            candidate_id=candidate_id,
            diff_summary=diff_summary[:12000],
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

    def _novelty_judge_prompt(
        self,
        *,
        candidate_json: str,
        registry_context_json: str,
        diff_summary: str,
    ) -> str:
        """生成候选方向去重审查提示词。"""
        return self._render_prompt(
            "novelty_judge",
            "judge",
            candidate_json=candidate_json,
            registry_context_json=registry_context_json,
            diff_summary=diff_summary,
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

    def summarize_candidate(
        self,
        *,
        candidate_id: str,
        diff_summary: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        """对已修改候选生成结构化字段；不允许修改文件。"""
        prompt = self._candidate_summary_prompt(
            candidate_id=candidate_id,
            diff_summary=diff_summary,
        )
        cmd = self._build_cmd(prompt, workdir, allowed_tools="Read")
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd,
            workdir,
            timeout_sec or min(self._timeout_sec, 120),
            render_title="Candidate Summarizer",
            transient_output=True,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def judge_novelty(
        self,
        *,
        candidate: dict[str, Any],
        registry_context: list[dict[str, Any]],
        diff_summary: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CodeAgentResult:
        """执行候选方向 novelty 去重审查。"""
        prompt = self._novelty_judge_prompt(
            candidate_json=json.dumps(candidate, ensure_ascii=False, indent=2),
            registry_context_json=json.dumps(
                registry_context,
                ensure_ascii=False,
                indent=2,
            ),
            diff_summary=diff_summary[:12000],
        )
        cmd = self._build_cmd(prompt, workdir, allowed_tools="Read")
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd,
            workdir,
            timeout_sec or min(self._timeout_sec, 240),
            render_title="Novelty Judge",
            transient_output=True,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)


class ClaudeCodeAgent(BaseCodeAgent):
    """Backed by Claude Code CLI (claude -p)."""

    backend_name = "claude"
    default_binary = "claude"
    supports_allowed_tools = True

    def __init__(
        self,
        model: str = "sonnet",
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        stream_output: bool = True,
        prompt_config_path: Path | None = None,
        binary: str | None = None,
    ):
        super().__init__(
            model=model,
            timeout_sec=timeout_sec,
            extra_args=extra_args,
            stream_output=stream_output,
            prompt_config_path=prompt_config_path,
            binary=binary,
        )

    def _extract_content(self, stdout: str) -> str:
        return _extract_content_from_stream_json(stdout)

    def _extract_error(self, stdout: str, stderr: str) -> str:
        effective_stderr = stderr.strip()
        if effective_stderr:
            return effective_stderr
        return _extract_error_from_stream_json(stdout)

    def _build_cmd(
        self,
        prompt: str,
        workdir: Path,
        *,
        allowed_tools: str = "Bash Edit Write Read",
    ) -> list[str]:
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
            allowed_tools,
            "--add-dir",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd


class CodexCodeAgent(BaseCodeAgent):
    """Backed by Codex CLI (codex exec)."""

    backend_name = "codex"
    default_binary = "codex"

    def __init__(
        self,
        model: str = "",
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        stream_output: bool = True,
        prompt_config_path: Path | None = None,
        binary: str | None = None,
        sandbox: str | None = None,
        approval_policy: str | None = None,
        ephemeral: bool | None = None,
    ):
        super().__init__(
            model=model,
            timeout_sec=timeout_sec,
            extra_args=extra_args,
            stream_output=stream_output,
            prompt_config_path=prompt_config_path,
            binary=binary,
        )
        self._sandbox = sandbox or os.getenv("AR_CODEX_SANDBOX", "danger-full-access")
        self._approval_policy = approval_policy or os.getenv(
            "AR_CODEX_APPROVAL_POLICY",
            "never",
        )
        if ephemeral is None:
            ephemeral = os.getenv("AR_CODEX_EPHEMERAL", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        self._ephemeral = ephemeral

    def _extract_content(self, stdout: str) -> str:
        return _extract_content_from_codex_json(stdout) or stdout.strip()

    def _build_cmd(
        self,
        prompt: str,
        workdir: Path,
        *,
        allowed_tools: str = "Bash Edit Write Read",
    ) -> list[str]:
        """组装 Codex CLI 命令行参数。"""
        _ = allowed_tools
        cmd = [
            self._binary,
            "--ask-for-approval",
            self._approval_policy,
            "exec",
            "--json",
            "-C",
            str(workdir),
            "--add-dir",
            str(workdir),
            "--sandbox",
            self._sandbox,
        ]
        if self._ephemeral:
            cmd.append("--ephemeral")
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        cmd.append(prompt)
        return cmd


def _split_extra_args(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


def make_code_agent(
    *,
    backend: str | None = None,
    model: str | None = None,
    timeout_sec: int = 600,
    stream_output: bool = True,
    prompt_config_path: Path | None = None,
) -> BaseCodeAgent:
    """按环境配置创建代码代理。"""
    selected = (backend or os.getenv("AR_AGENT_BACKEND", "claude")).strip().lower()
    selected = {
        "anthropic": "claude",
        "claude_code": "claude",
        "openai": "codex",
    }.get(selected, selected)

    if selected == "claude":
        agent_model = model if model is not None else os.getenv("AR_MODEL", "sonnet")
        return ClaudeCodeAgent(
            model=agent_model,
            timeout_sec=timeout_sec,
            extra_args=_split_extra_args(os.getenv("AR_CLAUDE_EXTRA_ARGS", "")),
            stream_output=stream_output,
            prompt_config_path=prompt_config_path,
            binary=os.getenv("AR_CLAUDE_BINARY") or None,
        )

    if selected == "codex":
        agent_model = model if model is not None else os.getenv("AR_CODEX_MODEL", "")
        return CodexCodeAgent(
            model=agent_model,
            timeout_sec=timeout_sec,
            extra_args=_split_extra_args(os.getenv("AR_CODEX_EXTRA_ARGS", "")),
            stream_output=stream_output,
            prompt_config_path=prompt_config_path,
            binary=os.getenv("AR_CODEX_BINARY") or None,
        )

    raise ValueError(
        f"Unsupported AR_AGENT_BACKEND={selected!r}; expected 'claude' or 'codex'"
    )
