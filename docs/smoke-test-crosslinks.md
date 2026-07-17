# Live smoke test — AGENTS.md + digest/scaffold cross-links — 2026-07-17

**Host:** `a DGX Spark workstation` (a DGX Spark workstation). **Server:** local vLLM on
`http://localhost:8000/v1` serving model id `qwen`
(root `bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid`, max_model_len 131072). No hosted
API keys were used. Branch under test: `agents-brief-crosslinks`.

**Artifact paths** (permanent, user-inspectable):

- Vault: `~/paperdigest-livetest/vault`
- Projects: `~/paperdigest-livetest/projects`
- Logs: `~/paperdigest-livetest/scaffold-retry.log`, `~/paperdigest-livetest/digest-rerun.log`

Four runs, in order: a scaffold attempt that failed on the default token cap, a
digest run with the project absent (missing-project path), a scaffold retest with
`--max-tokens 32768`, and a digest rerun with the project on disk (cross-link path).

## Run 1 — scaffold, default max_tokens (FAILED)

```bash
paperdigest scaffold 1706.03762 --dest ~/paperdigest-livetest/projects \
  --base-url http://localhost:8000/v1 --model qwen --vault ~/paperdigest-livetest/vault
```

Wall-clock: real 6m5s. Exit code: 1. Fail-loud, nothing written.

The run reached Stage 3/5 and died on the second stub:

```
Stage 3/5: writing stub 2/4: layers.py
cause: LLM call failed: OpenAI-compatible response truncated at max_tokens
```

The default `max_tokens` is 8192 and this model spends completion tokens on
thinking, so the visible output was truncated before the stub completed. The
debug dump at
`~/.cache/paperdigest/scaffold-debug-1706.03762-module-layers.py.txt`
contained "(no raw model output captured)".

## Run 2 — digest with the project NOT on disk (SUCCESS)

```bash
paperdigest 1706.03762 --vault ~/paperdigest-livetest/vault \
  --project-dir ~/paperdigest-livetest/projects \
  --base-url http://localhost:8000/v1 --model qwen
```

Run with `--project-dir` pointing at a directory in which the project did not yet
exist. Wall-clock: not recorded (the supervising agent died mid-run; the output
itself is complete and verified). Outcome: **SUCCESS**.

- `~/paperdigest-livetest/vault/Papers/2017-attention-is-all-you-need/` contains
  `00 Overview.md`, 8 concept notes, and `fig1.png`..`fig5.png` — all five figures
  were fetched and explained (the vision path worked).
- The Overview contains a "## Build it" section with the generate hint
  (`paperdigest scaffold 1706.03762 --dest ~/paperdigest-livetest/projects`).
- Zero concept notes contain a `*Build it:` line — the correct missing-project
  behavior.

## Run 3 — scaffold retest with --max-tokens 32768 (SUCCESS)

```bash
paperdigest scaffold 1706.03762 --dest ~/paperdigest-livetest/projects \
  --base-url http://localhost:8000/v1 --model qwen \
  --vault ~/paperdigest-livetest/vault --max-tokens 32768
```

Wall-clock: 16m14s (12:43:09 to 12:59:23 PDT). Outcome: **SUCCESS** — all 5
stages completed. Full log (verbatim):

```
Fetching arXiv 1706.03762...
Extracted 'Attention Is All You Need' (8 sections)
Stage 1/5: analyzing the paper...
Stage 2/5: planning modules...
Stage 3/5: writing stub 1/5: config.py
Stage 3/5: writing stub 2/5: attention.py
Stage 3/5: writing stub 3/5: layers.py
Stage 3/5: writing stub 4/5: embeddings.py
Stage 3/5: writing stub 5/5: transformer.py
Stage 3/5: writing tests/test_smoke.py
Stage 4/5: writing train/evaluate harness and configs...
Stage 5/5: writing README and EXPERIMENTS...
~/paperdigest-livetest/projects/2017-attention-is-all-you-need
```

Generated project verified at
`~/paperdigest-livetest/projects/2017-attention-is-all-you-need`: full
expected tree (AGENTS.md, README.md, EXPERIMENTS.md, train.py, evaluate.py,
configs/, src/, tests/, experiments/, .git). AGENTS.md is in the initial git
commit (`git ls-tree --name-only HEAD` lists it).

AGENTS.md "## Implementation order" (verbatim):

```
1. `config.py` — Defines default hyperparameters and configuration structure for the Transformer model. 1 TODO (§3.1, §7.1)
2. `attention.py` — Implements scaled dot-product attention and multi-head attention mechanisms. 2 TODOs (§3.2.1, §3.2.2)
3. `layers.py` — Implements position-wise feed-forward networks and sub-layer normalization with residual connections. 2 TODOs (§3.3, §3.1)
4. `embeddings.py` — Generates token embeddings and sinusoidal positional encodings for sequence ordering. 2 TODOs (§3.5, §3.4)
5. `transformer.py` — Constructs the encoder and decoder stacks and defines the complete Transformer model architecture. 20 TODOs (§3.1, §3.2.2, §3.3, §3.5, §3.2.3) — depends on: attention.py, layers.py, embeddings.py, config.py
```

AGENTS.md "## Paper notes" (verbatim):

```
Plain-English notes for this paper live in the Obsidian vault:
[`~/paperdigest-livetest/vault/Papers/2017-attention-is-all-you-need`](file://~/paperdigest-livetest/vault/Papers/2017-attention-is-all-you-need)
If that folder doesn't exist yet, generate it:
`paperdigest 1706.03762 --vault ~/paperdigest-livetest/vault`
```

The vault `file://` link is present and points at the real notes folder.

## Run 4 — digest rerun with the project on disk (SUCCESS, with one finding)

```bash
paperdigest 1706.03762 --vault ~/paperdigest-livetest/vault \
  --project-dir ~/paperdigest-livetest/projects \
  --base-url http://localhost:8000/v1 --model qwen --max-tokens 32768 --force
```

Wall-clock: 15m31s (13:02:36 to 13:18:07 PDT). Outcome: pipeline **SUCCESS**
(all 8 concepts, all 5 figures explained, 5 glossary terms, exit path printed
the vault folder). Last log lines (verbatim):

```
Explaining concept 8/8: Masked Self-Attention
Explaining figure 1/5: Figure 1: The Transformer - model...
Explaining figure 2/5: Figure 2: (left) Scaled Dot-Product Attention....
Explaining figure 3/5: Figure 3: An example of the...
Explaining figure 4/5: Figure 4: Two attention heads, also...
Explaining figure 5/5: Figure 5: Many of the attention...
Defining 5 glossary terms...
~/paperdigest-livetest/vault/Papers/2017-attention-is-all-you-need
```

Figures: `fig1.png`..`fig5.png` present in the vault folder; no figure warnings
in the log.

Overview "## Build it" correctly switched to the project-exists variant
(verbatim):

```
A scaffolded research project for this paper lives at
[`~/paperdigest-livetest/projects/2017-attention-is-all-you-need`](file://~/paperdigest-livetest/projects/2017-attention-is-all-you-need)
— open `AGENTS.md` there and hand it to a coding agent.
```

**Finding — concept-note cross-links did not appear.**
`grep -n "Build it:" vault/Papers/2017-attention-is-all-you-need/*.md` matched
nothing: zero of the 8 concept notes gained a `*Build it:` line, even though the
project existed on disk before the run started and section overlap is plainly
there. The concept notes' `*Source:*` lines (verbatim):

```
01 Self-Attention.md:      *Source: Section 2 of [Attention Is All You Need](...)*
02 Multi-Head Attention.md:*Source: Section 3.2.2 of [...]*
03 Scaled Dot-Product Attention.md: *Source: Section 3.2.1 of [...]*
04 Positional Encoding.md: *Source: Section 3.5 of [...]*
05 Encoder-Decoder Stack.md: *Source: Section 3.1 of [...]*
06 Residual Connections.md: *Source: Section 3.1 of [...]*
07 Position-wise Feed-Forward Networks.md: *Source: Section 3.3 of [...]*
08 Masked Self-Attention.md: *Source: Section 3.2.3 of [...]*
```

The project source cites the same sections — `grep -rn "§" .../src` shows, e.g.,
`attention.py: TODO(paper §3.2.1)` / `§3.2.2`, `layers.py: §3.3` / `§3.1`,
`embeddings.py: §3.5` / `§3.4`, `transformer.py: §3.1, §3.2.2, §3.2.3, §3.3,
§3.5`. So notes 02–08 each have at least one module citing their exact source
section, yet no cross-link was emitted. This is a section-matching bug (or the
matcher never ran), recorded here — not fixed as part of this test.

## Observations

- **Default `max_tokens=8192` is insufficient for this thinking model.** It spends
  completion tokens on thinking before the answer, so long stub generations
  truncate at the cap and the run fails loud (correctly). `--max-tokens 32768`
  is the workaround and both retests passed with it. Worth considering a larger
  default or a thinking-aware cap.
- **Vision figure explanations worked end-to-end** on both digest runs: 5/5
  figures fetched, saved as `figN.png` in the note folder, and explained, with
  no warnings.
- **The missing-project and project-exists Overview variants both behaved
  correctly**; the one broken piece is the per-concept `*Build it:` cross-link
  (see the finding in Run 4).
- Scaffold fail-loud behavior held in Run 1: exit 1, nothing written, debug dump
  path printed (though the dump itself contained "(no raw model output
  captured)" for the truncation case — a lesser observation worth a look).

## Retest after section-prefix fix (2026-07-17, commit 5a44bea)

The zero-concept-links finding above was diagnosed as a matching bug: the live
model emits concept headings like `Section 3.2.1`, but `_section_num` only
accepted digit-first headings (`3.2 Attention`), so every match failed. Fix:
optional case-insensitive `Section`/`Sec.`/`§` prefix in the regex, with
live-derived regression tests (23/23 render tests, 100% coverage on render.py).

- Command: same digest command as the rerun above (`--max-tokens 32768 --force`), exit 0, 18 progress lines.
- Outcome: **SUCCESS** — 5 of 8 concept notes now carry `*Build it: …*` module links:
  - `01 Transformer Architecture` → attention.py, config.py, embeddings.py, layers.py, transformer.py
  - `02 Scaled Dot-Product Attention` → attention.py
  - `03 Multi-Head Attention` → attention.py, transformer.py
  - `04 Positional Encoding` → embeddings.py, transformer.py
  - `05 Encoder and Decoder Stacks` → config.py, layers.py, transformer.py
- The 3 unlinked concepts cover sections no stub module cites — correct best-effort behavior.
- Concept titles differ from the earlier rerun (LLM nondeterminism across runs); links are consistent with each run's own sections.
