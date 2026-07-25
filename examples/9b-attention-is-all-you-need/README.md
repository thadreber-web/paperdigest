# Attention Is All You Need — digested & scaffolded by a local 9B

This is **real, unedited output** from a live end-to-end run of paperdigest
against "Attention Is All You Need" (arXiv:1706.03762), produced entirely by a
**local 9B model** with no cloud API. It is here as a worked example of what the
tool actually generates on modest, self-hostable hardware — the run logs are
included so the result is inspectable, not just asserted.

## The run

| | |
|---|---|
| **Paper** | Attention Is All You Need — arXiv:1706.03762 |
| **Model** | Qwen3.5-9B-Instruct (MTP, NVFP4 quant) — GGUF `qwen35-9b-mtp-nvfp4.gguf` (Hugging Face: `FreedomAISVR/Qwen3.5-9B-Instruct-MTP-NVFP4-GGUF`) |
| **Server** | llama.cpp (`llama-server`), OpenAI-compatible, `http://localhost:8001/v1`, 131072-token context |
| **Date** | 2026-07-25 (local) |
| **Result** | both pipelines exited 0 — see [`logs/`](logs/) |

[`logs/model-identity.txt`](logs/model-identity.txt) is an excerpt from the
server log showing exactly which model served the run;
[`logs/digest-final.log`](logs/digest-final.log) and
[`logs/scaffold-final.log`](logs/scaffold-final.log) are the stage-by-stage
run logs.

## What's inside

### `vault/` — `digest` output (Obsidian notes)
- `Papers/2017-attention-is-all-you-need/` — an overview note, 6 concept
  explainers, and 5 paper figures downloaded and explained by the model's
  vision path.
- `Glossary/` — 39 wikilinked jargon-term notes.

### `scaffold-project/` — `scaffold` output (research project skeleton)
A research-project layout with 4 paper-derived module stubs (`attention.py`,
`layers.py`, `position.py`, `transformer.py`) carrying cited `§` section
references, a smoke test, a `train.py`/`evaluate.py` harness, `configs/`,
`EXPERIMENTS.md` (targets cited to the paper's own tables, not fabricated
results), and a generated `AGENTS.md` brief for handing the project to a coding
agent. The scaffolder normally initializes a git repo inside the project; that
nested `.git` was removed so this copy commits cleanly into paperdigest.

## Reproduce it

With a llama.cpp server running your 9B on port 8001:

```bash
# plain-English Obsidian notes
paperdigest digest 1706.03762 --vault ~/ObsidianVault --base-url http://localhost:8001/v1

# research project skeleton
paperdigest scaffold 1706.03762 --dest ~/projects --base-url http://localhost:8001/v1
```

llama.cpp ignores `--model`; on Ollama/vLLM add `--model <name>`. Output varies
run to run — the model improvises wording, and small models occasionally need a
retry — so a fresh run will not be byte-identical to this one. This folder is
one real, complete result.
