"""Mermaid diagram post-processing.

Two layers, mirroring the JSON-robustness approach: a deterministic sanitizer
(auto-quote node labels containing special characters — the most common way
small local models break Mermaid) that always runs, and an optional parse
validation that activates only when a local node + mermaid install is
detected. No node, no problem: validation is skipped, the sanitized diagram
is kept as-is, and Obsidian renders it (or shows an error box for the rare
residual mistake).
"""
from __future__ import annotations

import re
import shutil
import subprocess

_LABEL_RE = re.compile(r"\[([^\[\]\n]+)\]")
_SPECIAL_RE = re.compile(r"[(){}<>:,]")
_FENCE_RE = re.compile(r"(```mermaid[ \t]*\r?\n)(.*?)(\r?\n```)", re.DOTALL)

_available: bool | None = None  # None = not yet detected; cached per process

_PARSE_JS = """\
import "global-jsdom/register";
const mermaid = (await import("mermaid")).default;
const chunks = [];
process.stdin.on("data", (d) => chunks.push(d));
process.stdin.on("end", async () => {
  try {
    await mermaid.parse(chunks.join(""));
    console.log("PARSE_OK");
  } catch {
    console.log("PARSE_FAIL");
  }
});
"""


def _quote_label(m: re.Match) -> str:
    inner = m.group(1)
    if inner.startswith('"') or not _SPECIAL_RE.search(inner):
        return m.group(0)
    return '["' + inner.replace('"', "'") + '"]'


def sanitize_block(code: str) -> str:
    """Quote node labels that contain characters Mermaid's parser chokes on."""
    return _LABEL_RE.sub(_quote_label, code)


def sanitize_markdown(md: str) -> str:
    """Apply sanitize_block to every ```mermaid fence in a markdown document."""
    return _FENCE_RE.sub(lambda m: m.group(1) + sanitize_block(m.group(2).replace("\r\n", "\n")) + m.group(3), md)


def mermaid_blocks(md: str) -> list[str]:
    return [m.group(2).replace("\r\n", "\n") for m in _FENCE_RE.finditer(md)]


def _run_parser(code: str) -> str | None:
    try:
        result = subprocess.run(
            ["node", "--input-type=module", "-e", _PARSE_JS],
            input=code, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = result.stdout.strip()
    return out if out in ("PARSE_OK", "PARSE_FAIL") else None


def validator_available() -> bool:
    """Detect (once per process) whether node + the mermaid npm package work here."""
    global _available
    if _available is None:
        if shutil.which("node") is None:
            _available = False
        else:
            _available = _run_parser("flowchart TD\n    A --> B") == "PARSE_OK"
    return _available


def validate(code: str) -> bool | None:
    """True/False if the local parser is available, None if it isn't."""
    if not validator_available():
        return None
    result = _run_parser(code)
    return None if result is None else result == "PARSE_OK"
