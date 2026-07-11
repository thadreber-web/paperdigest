# Live smoke test — 2026-07-11

**Host:** spark-8802 (NVIDIA GB10). **Backend:** `openai` pointed at a local
llama.cpp server on `http://127.0.0.1:8001/v1` serving
`FreedomAISVR/Qwen3.5-9B-Instruct-MTP-NVFP4-GGUF` (32k context, full GPU
offload). No hosted API keys were used.

## Command

```bash
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 \
  --vault /tmp/smoke-vault --level intermediate \
  --backend openai --model qwen35-9b --base-url http://127.0.0.1:8001/v1
```

Exit 0. Progress on stderr, folder path on stdout:

```
Fetching arXiv 1706.03762...
Extracted 'Attention Is All You Need' (8 sections)
Outlining 'Attention Is All You Need'...
Explaining concept 1/7: Self-Attention Mechanism
...
Explaining concept 7/7: Parallelization Advantage
Defining 24 glossary terms...
/tmp/smoke-vault/Papers/2017-attention-is-all-you-need
```

## Artifacts verified (by reading them)

- `00 Overview.md`: frontmatter (`source:`, `model: qwen35-9b`,
  `level: intermediate`), TLDR, 7 concept wikilinks (`[[01 Self-Attention
  Mechanism]]` …), 24 glossary wikilinks, 5 self-test questions.
- All 7 concept notes contain fenced blocks; `03 Encoder-Decoder
  Architecture.md` has a genuine ASCII box diagram of the encoder/decoder
  stacks; `01 Self-Attention Mechanism.md` walks Eq. 1 line by line:
  > **Scaling ($\frac{1}{\sqrt{d_k}}$):** … Without this scaling, if the
  > dimension $d_k$ is large, the dot products get huge numbers.
- 24 notes in `Glossary/`, e.g. `softmax.md`:
  > A function that converts a vector of raw scores into a probability
  > distribution where all values are non-negative and sum to one …

## Re-run guard (live)

Second run without `--force` exited 1 with:

```
error: /tmp/smoke-vault/Papers/2017-attention-is-all-you-need already exists — re-run with --force to overwrite (your edits there will be lost)
```

On the re-run, only 10 glossary terms were sent to the LLM (existing term
notes were correctly excluded from definition requests).

## Observations (not fixed; noted for later)

- The existing-folder check happens at render time, so a blocked re-run still
  spends the full set of LLM calls before failing. Checking earlier would
  save tokens/time.
- Concept outputs from the model can repeat the note title as a second `#`
  heading under the rendered one (cosmetic).
- Small models fail honestly: `qwen2.5:3b` via a CPU-only, 4096-context
  Ollama snap produced structured `error: ... unparseable JSON` / `missing
  the 'tldr' field` messages with exit 1 — the error paths work as designed.
