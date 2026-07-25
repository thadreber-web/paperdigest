"""Zero-dependency config loader: flat `key: value` files with typed values."""
from __future__ import annotations

import ast
from pathlib import Path


def load_config(path):
    """Read a flat config file into a dict of typed values.

    One entry per line as `key: value` (YAML) or `key = value` (.env/.ini); the first
    `:` or `=` on the line separates key from value. Blank lines and `#` comments are
    ignored. Values are read with Python literal rules (int, float including `1e-9`,
    list, quoted string) plus lowercase `true`/`false`; anything else stays a string.
    """
    config = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        seps = [i for i in (line.find(":"), line.find("=")) if i != -1]
        if not seps:
            continue
        i = min(seps)
        key, val = line[:i].strip(), line[i + 1:].strip()
        low = val.lower()
        if low in ("true", "false"):
            config[key] = low == "true"
            continue
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            config[key] = val
    return config
