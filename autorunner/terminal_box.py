"""Rich-based terminal box renderer for streaming agent output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any


_MAX_STREAM_DISPLAY_CHARS = 1000
_MAX_EXPANDED_DISPLAY_CHARS = 5000
_MAX_COLLAPSED_PREVIEW_CHARS = 80
_MAX_STORED_SECTION_CHARS = 200000


@dataclass
class StreamRenderConfig:
    title: str
    max_lines: int = 300
    enabled: bool = True
    show_thinking: bool = True
    full_output_path: str | None = None
    transient: bool = False


class AgentOutputBox:
    """Render streaming stdout/stderr in a terminal panel."""

    def __init__(self, config: StreamRenderConfig):
        self._config = config
        self._stdout_lines: deque[str] = deque(maxlen=max(20, config.max_lines))
        self._stderr_lines: deque[str] = deque(maxlen=max(20, config.max_lines // 2))
        self._thinking_lines: deque[str] = deque(maxlen=max(20, config.max_lines // 2))
        self._tool_lines: deque[str] = deque(maxlen=max(20, config.max_lines // 2))
        self._answer_lines: deque[str] = deque(maxlen=max(20, config.max_lines))
        self._ordered_blocks: deque[dict[str, Any]] = deque(
            maxlen=max(30, config.max_lines)
        )
        self._thinking_content = ""
        self._tool_content = ""
        self._answer_content = ""
        self._status_line = ""
        self._thinking_dropped = 0
        self._tool_dropped = 0
        self._answer_dropped = 0
        self._thinking_char_dropped = 0
        self._tool_char_dropped = 0
        self._answer_char_dropped = 0
        self._usage_line = ""
        self._start_time = time.monotonic()
        self._elapsed_seconds = 0
        self._is_active = True
        self._thinking_collapsed = False
        self._tool_collapsed = False
        self._answer_collapsed = False
        self._available = False
        self._console = None
        self._live = None

        if not config.enabled:
            return

        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            self._live = Live(
                console=self._console,
                refresh_per_second=8,
                transient=self._config.transient,
            )
            self._available = True
        except Exception:
            self._available = False

    def __enter__(self) -> "AgentOutputBox":
        if self._available and self._live is not None:
            self._live.__enter__()
            self._refresh()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._available and self._live is not None:
            self.finalize()
            # Render one final frame with Markdown for better readability.
            self._refresh(render_markdown=True)
            self._live.__exit__(exc_type, exc, tb)

    def add_stdout(self, line: str):
        text = line.rstrip("\n")
        if not text:
            return
        self._stdout_lines.append(text)
        self._refresh()

    def add_stderr(self, line: str):
        text = line.rstrip("\n")
        if not text:
            return
        self._stderr_lines.append(text)
        self._refresh()

    def add_thinking(self, text: str):
        if not text:
            return
        self._append_content("thinking", text)
        self._append_delta("thinking", self._thinking_lines, text)
        self._refresh()

    def add_text(self, text: str):
        if not text:
            return
        self._append_content("answer", text)
        self._append_delta("answer", self._answer_lines, text)
        self._refresh()

    def add_tool_call(self, text: str):
        if not text:
            return
        self._append_content("tool", text)
        self._append_delta("tool", self._tool_lines, text)
        self._refresh()

    def add_ordered_block(self, kind: str, text: str, *, render_markdown: bool = False):
        if not text:
            return
        kind_l = kind.lower()
        if kind_l == "thinking":
            title = "Thinking"
            border_style = "blue"
            dim = True
        elif kind_l == "assistant":
            title = "Assistant"
            border_style = "green"
            dim = False
        else:
            title = "Tools"
            border_style = "yellow"
            dim = False

        self._ordered_blocks.append(
            {
                "kind": kind_l,
                "title": title,
                "border_style": border_style,
                "content": text,
                "dim": dim,
                "render_markdown": render_markdown,
            }
        )
        self._refresh()

    def _clear_previous_thinking_and_tools(self):
        if not self._ordered_blocks:
            return

        kept: deque[dict[str, Any]] = deque(maxlen=self._ordered_blocks.maxlen)
        for block in self._ordered_blocks:
            kind = str(block.get("kind") or "").lower()
            title = str(block.get("title") or "").lower()
            is_tool_like = kind in {"tools", "tools-inline", "tool"} or title == "tools"
            is_thinking_like = kind == "thinking" or title == "thinking"
            if is_tool_like or is_thinking_like:
                continue
            kept.append(block)

        self._ordered_blocks = kept

    def _clear_previous_assistant_blocks(self):
        if not self._ordered_blocks:
            return

        kept: deque[dict[str, Any]] = deque(maxlen=self._ordered_blocks.maxlen)
        for block in self._ordered_blocks:
            kind = str(block.get("kind") or "").lower()
            title = str(block.get("title") or "").lower()
            is_assistant_like = kind == "assistant" or title == "assistant"
            if is_assistant_like:
                continue
            kept.append(block)

        self._ordered_blocks = kept

    def add_ordered_thinking(self, text: str):
        self._clear_previous_thinking_and_tools()
        self.add_ordered_block("thinking", text)

    def add_ordered_tools(self, text: str):
        self.add_ordered_block("tools", text)

    def add_ordered_tools_renderable(self, renderable: Any):
        if renderable is None:
            return
        self._ordered_blocks.append(
            {
                "kind": "tools-inline",
                "renderable": renderable,
            }
        )
        self._refresh()

    def add_ordered_assistant(self, text: str, *, stream_id: str | None = None):
        if not text:
            return

        def _normalize_chunk(chunk: str) -> str:
            # Stream deltas often start with stray newlines and sentence-per-line breaks.
            cleaned = chunk.replace("\r", "").lstrip("\n")
            return cleaned.replace("\n", " ")

        normalized = _normalize_chunk(text)
        if not normalized:
            return

        # Keep chunks from the same assistant stream in one box.
        if self._ordered_blocks:
            last = self._ordered_blocks[-1]
            last_kind = str(last.get("kind") or "").lower()
            last_stream_id = str(last.get("stream_id") or "")
            if (
                last_kind == "assistant"
                and stream_id
                and last_stream_id
                and last_stream_id == stream_id
            ):
                prev = str(last.get("content") or "")
                if prev and not prev[-1].isspace() and not normalized[0].isspace():
                    last["content"] = prev + " " + normalized
                else:
                    last["content"] = prev + normalized
                self._refresh()
                return

            self._clear_previous_assistant_blocks()
        self.add_ordered_block("assistant", normalized)
        if stream_id and self._ordered_blocks:
            self._ordered_blocks[-1]["stream_id"] = stream_id

    def finalize(self):
        """Mark stream as complete and collapse long sections by default."""
        self._is_active = False
        self._thinking_collapsed = bool(self._thinking_content.strip())
        self._tool_collapsed = bool(self._tool_content.strip())
        self._answer_collapsed = (
            len(self._answer_content.strip()) > _MAX_STREAM_DISPLAY_CHARS
        )
        self._refresh()

    def expand_thinking(self):
        self._thinking_collapsed = False
        self._refresh()

    def collapse_thinking(self):
        self._thinking_collapsed = True
        self._refresh()

    def toggle_thinking(self):
        self._thinking_collapsed = not self._thinking_collapsed
        self._refresh()

    def expand_tools(self):
        self._tool_collapsed = False
        self._refresh()

    def collapse_tools(self):
        self._tool_collapsed = True
        self._refresh()

    def toggle_tools(self):
        self._tool_collapsed = not self._tool_collapsed
        self._refresh()

    def expand_answer(self):
        self._answer_collapsed = False
        self._refresh()

    def collapse_answer(self):
        self._answer_collapsed = True
        self._refresh()

    def toggle_answer(self):
        self._answer_collapsed = not self._answer_collapsed
        self._refresh()

    def set_usage(self, usage: dict | None, total_cost_usd: float | None = None):
        if not usage and total_cost_usd is None:
            return

        usage = usage or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        cache_tok = usage.get("cache_read_input_tokens")
        if cache_tok is None:
            cache_tok = usage.get("cached_input_tokens")

        parts = []
        if in_tok is not None:
            parts.append(f"in={in_tok}")
        if out_tok is not None:
            parts.append(f"out={out_tok}")
        if cache_tok is not None:
            parts.append(f"cache={cache_tok}")
        if total_cost_usd is not None:
            parts.append(f"cost=${total_cost_usd:.6f}")

        if parts:
            self._usage_line = "tokens: " + " | ".join(parts)
            self._refresh()

    def _append_delta(self, section: str, lines: deque[str], text: str):
        """Append stream delta text without breaking sentence chunks."""
        normalized = text.replace("\r", "")
        parts = normalized.split("\n")

        if not lines:
            lines.append("")

        # First segment continues current line. For punctuation boundary, insert a space.
        current = lines[-1]
        first = parts[0]
        if (
            current
            and first
            and not current[-1].isspace()
            and not first[0].isspace()
            and current[-1] in ".!?;:，。！？；：)】]"
        ):
            lines[-1] = f"{current} {first}"
        else:
            lines[-1] = f"{current}{first}"

        # Remaining segments start new lines.
        for part in parts[1:]:
            if len(lines) == lines.maxlen:
                if section == "thinking":
                    self._thinking_dropped += 1
                elif section == "tool":
                    self._tool_dropped += 1
                else:
                    self._answer_dropped += 1
            lines.append(part)

    def _append_content(self, section: str, text: str):
        normalized = text.replace("\r", "")
        if section == "thinking":
            merged = self._thinking_content + normalized
            if len(merged) > _MAX_STORED_SECTION_CHARS:
                overflow = len(merged) - _MAX_STORED_SECTION_CHARS
                self._thinking_char_dropped += overflow
                merged = merged[-_MAX_STORED_SECTION_CHARS:]
            self._thinking_content = merged
            return

        if section == "tool":
            merged = self._tool_content + normalized
            if len(merged) > _MAX_STORED_SECTION_CHARS:
                overflow = len(merged) - _MAX_STORED_SECTION_CHARS
                self._tool_char_dropped += overflow
                merged = merged[-_MAX_STORED_SECTION_CHARS:]
            self._tool_content = merged
            return

        merged = self._answer_content + normalized
        if len(merged) > _MAX_STORED_SECTION_CHARS:
            overflow = len(merged) - _MAX_STORED_SECTION_CHARS
            self._answer_char_dropped += overflow
            merged = merged[-_MAX_STORED_SECTION_CHARS:]
        self._answer_content = merged

    def _char_count_label(self, size: int) -> str:
        if size >= 1000:
            return f"{size / 1000:.1f}k chars"
        return f"{size:,} chars"

    def _tail_truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return "..." + text[-limit:]

    def _middle_elide(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + "\n\n... (truncated) ...\n\n" + text[-half:]

    def _first_line_preview(
        self, text: str, limit: int = _MAX_COLLAPSED_PREVIEW_CHARS
    ) -> str:
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line) <= limit:
            return first_line
        return first_line[: limit - 1] + "..."

    def _build_section_panel(
        self,
        *,
        title: str,
        border_style: str,
        content: str,
        collapsed: bool,
        expand_hint: str,
        dim: bool = False,
        render_markdown: bool = False,
    ):
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text

        label = f"{title} ({self._char_count_label(len(content))})"
        if self._is_active:
            body_text = self._tail_truncate(content, _MAX_STREAM_DISPLAY_CHARS)
            body = Text(body_text, style="dim" if dim else "")
        elif collapsed:
            preview = Text(self._first_line_preview(content), style="dim")
            preview.append(f"  [{expand_hint} 可展开]", style="dim italic")
            body = preview
        else:
            expanded = self._middle_elide(content, _MAX_EXPANDED_DISPLAY_CHARS)
            if render_markdown:
                body = Markdown(expanded)
            else:
                body = Text(expanded, style="dim" if dim else "")

        return Panel(body, title=label, border_style=border_style, padding=(0, 1))

    def set_status(self, status: str):
        self._status_line = status.strip()
        self._refresh()

    def update_elapsed(self, seconds: int):
        if seconds < 0:
            seconds = 0
        if self._elapsed_seconds == seconds:
            return
        self._elapsed_seconds = seconds
        self._refresh()

    def _refresh(self, render_markdown: bool = False):
        if not self._available or self._live is None:
            return

        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        stdout_text = "\n".join(self._stdout_lines)
        answer_text = self._answer_content.rstrip()
        tool_text = self._tool_content.rstrip()
        thinking_text = self._thinking_content.rstrip()
        fallback = stdout_text if stdout_text else "(waiting for output...)"

        if self._config.show_thinking:
            thinking_seconds = max(
                self._elapsed_seconds, int(time.monotonic() - self._start_time)
            )
            header = f"[dim]Thinking... ({thinking_seconds}s)[/dim]"
        else:
            header = ""

        meta_sections: list[object] = []
        if header:
            meta_sections.append(Text.from_markup(header))
        if self._status_line:
            meta_sections.append(
                Text.from_markup(f"[bold cyan]Status:[/bold cyan] {self._status_line}")
            )
        if self._usage_line:
            meta_sections.append(
                Text.from_markup(
                    f"[bold magenta]Usage:[/bold magenta] {self._usage_line}"
                )
            )
        renderables: list[object] = []
        if meta_sections:
            renderables.append(
                Panel(
                    Group(*meta_sections), title=self._config.title, border_style="cyan"
                )
            )

        if self._ordered_blocks:
            for block in self._ordered_blocks:
                if block.get("kind") == "tools-inline":
                    renderable = block.get("renderable")
                    if renderable is not None:
                        renderables.append(renderable)
                    continue

                content = str(block.get("content") or "").rstrip()
                if not content:
                    continue
                renderables.append(
                    self._build_section_panel(
                        title=str(block.get("title") or "Section"),
                        border_style=str(block.get("border_style") or "cyan"),
                        content=content,
                        collapsed=False,
                        expand_hint="",
                        dim=bool(block.get("dim", False)),
                        render_markdown=(
                            bool(block.get("render_markdown", False))
                            and render_markdown
                        ),
                    )
                )
        else:
            if thinking_text:
                renderables.append(
                    self._build_section_panel(
                        title="Thinking",
                        border_style="blue",
                        content=thinking_text,
                        collapsed=self._thinking_collapsed,
                        expand_hint="expand_thinking()",
                        dim=True,
                    )
                )

            if tool_text:
                renderables.append(
                    self._build_section_panel(
                        title="Tools",
                        border_style="yellow",
                        content=tool_text,
                        collapsed=self._tool_collapsed,
                        expand_hint="expand_tools()",
                    )
                )

            if answer_text:
                renderables.append(
                    self._build_section_panel(
                        title="Assistant",
                        border_style="green",
                        content=answer_text,
                        collapsed=self._answer_collapsed,
                        expand_hint="expand_answer()",
                        render_markdown=render_markdown,
                    )
                )

        if not self._ordered_blocks:
            dropped_total = (
                self._thinking_dropped + self._tool_dropped + self._answer_dropped
            )
            if dropped_total > 0:
                note = f"... 已折叠 {dropped_total} 行"
                if self._config.full_output_path:
                    link = self._config.full_output_path
                    note += f"，完整内容见 [link=file://{link}]{link}[/link]"
                renderables.append(
                    Panel(
                        Text.from_markup(f"[dim]{note}[/dim]"),
                        border_style="bright_black",
                    )
                )

            dropped_chars = (
                self._thinking_char_dropped
                + self._tool_char_dropped
                + self._answer_char_dropped
            )
            if dropped_chars > 0:
                renderables.append(
                    Panel(
                        Text.from_markup(
                            f"[dim]... 已丢弃 {dropped_chars} chars（超出内存窗口）[/dim]"
                        ),
                        border_style="bright_black",
                    )
                )

        if self._stderr_lines:
            stderr_text = "\n".join(self._stderr_lines)
            renderables.append(
                Panel(
                    Text(stderr_text),
                    title="stderr",
                    border_style="red",
                    padding=(0, 1),
                )
            )

        if not renderables:
            renderables.append(
                Panel(Text(fallback), title=self._config.title, border_style="cyan")
            )

        self._live.update(Group(*renderables))
