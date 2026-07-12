# Live smoke test — scaffold subcommand — 2026-07-12

**Host:** spark-8802 (NVIDIA GB10). **Server:** local llama.cpp server on
`http://localhost:8001/v1` serving
`~/models/Qwythos-9B-v2-NVFP4/Qwythos-9B-v2-MTP-NVFP4.gguf` (confirmed via
`curl -s http://localhost:8001/v1/models` before the run). No hosted API keys
were used; no model server was started for this test.

## Command

```bash
.venv/bin/paperdigest scaffold 1706.03762 --dest /tmp/claude-scaffold-smoke --base-url http://localhost:8001/v1
```

Wall-clock: ~1m26s (16:42:10 UTC to 16:43:36 UTC). Exit code: 1.

## Outcome

The run reached Stage 2/5 ("planning modules") and then failed with a loud
ERROR banner — a valid, recordable outcome per the design (no auto-fallback).
Nothing was written to the destination directory; `/tmp/claude-scaffold-smoke`
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
  raw model output saved to: ~/code/paperdigest/scaffold-debug-plan.txt
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
