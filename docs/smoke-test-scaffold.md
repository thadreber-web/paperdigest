# Live smoke test — scaffold subcommand — 2026-07-12

**Host:** a DGX Spark workstation (NVIDIA GB10). **Server:** local llama.cpp server on
`http://localhost:8001/v1` serving
`Qwythos-9B-v2-MTP-NVFP4.gguf` (confirmed via
`curl -s http://localhost:8001/v1/models` before the run). No hosted API keys
were used; no model server was started for this test.

## Command

```bash
.venv/bin/paperdigest scaffold 1706.03762 --dest /tmp/scaffold-smoke --base-url http://localhost:8001/v1
```

Wall-clock: ~1m26s (16:42:10 UTC to 16:43:36 UTC). Exit code: 1.

## Outcome

The run reached Stage 2/5 ("planning modules") and then failed with a loud
ERROR banner — a valid, recordable outcome per the design (no auto-fallback).
Nothing was written to the destination directory; `/tmp/scaffold-smoke`
does not exist after the run.

stderr (verbatim):

```
Fetching arXiv 1706.03762...
Extracted 'Attention Is All You Need' (8 sections)
Stage 1/5: analyzing the paper...
Stage 2/5: planning modules...
================================================================
ERROR: scaffold stage failed — aborting, nothing was written
  stage:  plan
  model:  qwen35-9b
  server: http://localhost:8001/v1
  cause:  model returned invalid JSON: Extra data: line 1 column 1633 (char 1632)
  raw model output saved to: <repo>/scaffold-debug-plan.txt
================================================================
```

stdout was empty.

## Stage-by-stage

- **Stage 1 (analyze):** succeeded — no error reported, progressed to Stage 2.
- **Stage 2 (plan):** failed. The raw model output saved to
  `scaffold-debug-plan.txt` is well-formed JSON describing 5 modules
  (`layers.py`, `encoder.py`, `decoder.py`, `model.py`, `utils.py`) with
  responsibilities, API signatures, and dependencies — but the model appended
  a single stray backtick (`` ` ``) after the final closing `}`, e.g.:

  ```
  ...["model.py"]}]}`
  ```

  That trailing character is what `json.loads` / the parser flagged as
  "Extra data: line 1 column 1633 (char 1632)" — the JSON payload itself was
  otherwise complete and well-structured.
- **Stages 3-5:** not reached.

## Re-run guard / dest-dir state

Not exercised — the run failed before any directory was created, so the
existing-folder / `--force` guard was not triggered in this test.

## Observations (noted for later, not fixed as part of this task)

- The 9B model produced substantively correct, well-formed JSON for the plan
  stage but appended one extraneous character (a stray backtick, likely a
  markdown code-fence remnant) that broke strict JSON parsing. This is
  consistent with small/mid-size instruct models occasionally leaking
  formatting tokens into structured output. The scaffold pipeline's
  fail-loud/no-partial-write behavior worked exactly as designed: it aborted
  cleanly, wrote nothing to the destination, and preserved the raw output for
  inspection.

## Retest after JSON-robustness fixes (2026-07-12, commit 051849a)

Two fixes were applied after the failed run above:
`response_format={"type": "json_object"}` is now requested from
OpenAI-compatible servers for JSON stages (with a sticky fallback if the
server rejects it), and `_stage_json` parses via `json.JSONDecoder().raw_decode`,
which reads the first complete JSON value and ignores trailing junk such as
the stray backtick that killed the first run.

- Model/server: same as above — Qwythos-9B-v2-MTP-NVFP4 (llama.cpp, `http://localhost:8001/v1`)
- Paper: arXiv 1706.03762 ("Attention Is All You Need", 8 sections)
- Command: `paperdigest scaffold 1706.03762 --dest <tmp> --base-url http://localhost:8001/v1`
- Outcome: **SUCCESS**, exit 0, wall clock 7m24s (`time`: real 7m24.341s)
- All 5 stages completed; the plan stage that previously failed passed.
- Generated project verified: full expected tree (4 stub modules: encoder,
  decoder, attention, positioning_encoding), one git commit
  ("Scaffold generated from arXiv:1706.03762 by paperdigest"), all 9 Python
  files parse with `ast.parse`, `configs/base.yaml` contains the paper's real
  hyperparameters (N: 6, d_model: 512, d_ff: 2048, h: 8, d_k: 64, d_v: 64).

Observations:

- The model fully implemented `attention.py` and `encoder.py` (torch code
  with paper-section docstrings) rather than leaving TODO stubs; only
  `decoder.py` and `positioning_encoding.py` contain `TODO(paper §…)`
  markers. The prompt allows implementing "simple glue", and the model
  stretched that; output is still valid, but stub discipline varies at 9B.
- Module filename `positioning_encoding.py` (sic) — the model's own naming;
  harmless.
