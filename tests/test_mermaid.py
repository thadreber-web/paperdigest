from paperdigest import mermaid


def test_sanitize_quotes_labels_with_special_chars():
    code = "flowchart LR\n    A[Input (x1..xn)] --> B[Encoder Stack (6 layers)]\n"
    fixed = mermaid.sanitize_block(code)
    assert 'A["Input (x1..xn)"]' in fixed
    assert 'B["Encoder Stack (6 layers)"]' in fixed


def test_sanitize_leaves_quoted_and_simple_labels_alone():
    code = 'flowchart TD\n    A["already (quoted)"] --> B[Plain label]\n'
    assert mermaid.sanitize_block(code) == code


def test_sanitize_markdown_only_touches_mermaid_fences():
    md = (
        "Some text with [a link](x) and brackets [not (a) diagram].\n\n"
        "```mermaid\nflowchart LR\n    A[Foo (bar)] --> B[Ok]\n```\n\n"
        "```python\nx = arr[foo(1)]\n```\n"
    )
    out = mermaid.sanitize_markdown(md)
    assert '[not (a) diagram]' in out          # prose untouched
    assert 'x = arr[foo(1)]' in out            # other fences untouched
    assert 'A["Foo (bar)"]' in out             # mermaid fence fixed


def test_sanitize_replaces_double_quotes_inside_labels():
    code = 'flowchart LR\n    A[Say "hi" (loudly)] --> B[Ok]\n'
    fixed = mermaid.sanitize_block(code)
    assert "A[\"Say 'hi' (loudly)\"]" in fixed


def test_validator_unavailable_without_node(monkeypatch):
    monkeypatch.setattr(mermaid, "_available", None)  # reset detection cache
    monkeypatch.setattr(mermaid.shutil, "which", lambda cmd: None)
    assert mermaid.validator_available() is False
    assert mermaid.validate("flowchart LR\n    A --> B") is None


def test_validator_detection_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(mermaid, "_available", None)
    monkeypatch.setattr(mermaid.shutil, "which", lambda cmd: calls.append(cmd) or None)
    mermaid.validator_available()
    mermaid.validator_available()
    assert len(calls) == 1


def test_sanitize_markdown_handles_crlf_fence():
    md = (
        "```mermaid\r\n"
        "flowchart LR\r\n"
        "    A[Foo (bar)] --> B[Ok]\r\n"
        "```\r\n"
    )
    out = mermaid.sanitize_markdown(md)
    assert 'A["Foo (bar)"]' in out
    body = out.split("```mermaid")[1].split("```")[0].strip("\r\n")
    assert "\r" not in body


def test_mermaid_blocks_handles_crlf_and_trailing_space_fence():
    md = (
        "```mermaid  \r\n"
        "flowchart LR\r\n"
        "    A --> B\r\n"
        "```\n"
    )
    blocks = mermaid.mermaid_blocks(md)
    assert len(blocks) == 1
    assert "\r" not in blocks[0]
    assert "flowchart LR\n    A --> B" == blocks[0]


def test_validate_parses_probe_output(monkeypatch):
    monkeypatch.setattr(mermaid, "_available", True)

    class _Result:
        stdout = "PARSE_FAIL\n"

    monkeypatch.setattr(mermaid.subprocess, "run", lambda *a, **k: _Result())
    assert mermaid.validate("flowchart LR\n    A[Bad (label)] --> B") is False
