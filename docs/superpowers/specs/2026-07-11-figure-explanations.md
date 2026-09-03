# paperdigest — Figure Explanations Spec

Date: 2026-07-11
Status: Implemented (shipped 2026-07-25 release; see CHANGELOG). Spec kept for provenance.

## Purpose

Papers' figures are often their highest-information content, and v1 drops
them entirely. This feature extracts figure images from the paper HTML, has
a vision-capable model explain each one at the configured level, and renders
image + explanation side by side in the paper's note folder.

Local-first: the reference deployment is llama.cpp with an `--mmproj`
projector loaded (verified working on this machine: Qwen3.5-9B on port
8001). Cloud backends with vision (Anthropic) get parity; backends without
vision skip gracefully.

## Feasibility evidence (sanity check, 2026-07-11)

Sent Figure 1 of "Attention Is All You Need" (ModalNet-21.png, 1520×2239)
to the local llama-server via the standard OpenAI multimodal message format
through the existing `/v1/chat/completions` endpoint — no server or SDK
changes needed.

- **Image alone, no context:** perception was accurate (read every label:
  Nx stacks, Masked Multi-Head Attention, Add & Norm, "shifted right",
  Linear → Softmax, cross-attention arrows) but interpretation spiraled into
  ~1,500 words of self-correction and misattributed the figure to BART.
- **Image + caption + paper TLDR + 250-word cap:** tight, accurate,
  level-appropriate explanation. Minor errors only (placed positional
  encoding on one side; omitted the cross-attention sublayer).

Conclusion: the model's eyes are good; its unsupervised judgment is not.
Every design choice below follows from that.

## Prompt guidelines (normative)

These are requirements, not suggestions — each one prevents a failure mode
observed in the sanity check:

1. **Always send context with the image.** The user message MUST include the
   paper title, the TLDR (available from the outline step by the time
   figures are processed), and the figure's own caption verbatim. A naked
   image invites confabulated paper identification.
2. **Describe, don't identify.** Ask "explain what this figure shows in this
   paper and why it matters," never "what is this figure / where is it
   from." The paper identity is given, not inferred.
3. **Hard word cap in the system prompt** (~250 words). Uncapped, the model
   rambles and second-guesses; capped, it stays disciplined.
4. **Uncertainty escape hatch.** System prompt MUST include: "If any detail
   is unclear in the image, say so instead of guessing." Confidently wrong
   axis readings are worse than no explanation.
5. **Level-aware.** Reuse the existing beginner/intermediate/advanced
   phrasing from the concept prompts so figure notes match the rest of the
   digest.
6. **Prose output, not JSON.** Figure explanations are free text like
   concept bodies — no structured-output parsing to fail. One figure per
   call (images eat context; a figure + prompt fits a 32k window, batches
   may not).

Reference system prompt (validated in the sanity check, adjust with care):

> You explain figures from research papers to a {level}-level learner. Be
> concise: at most 250 words. Explain what the figure shows and why it
> matters for the paper. If any detail is unclear in the image, say so
> instead of guessing.

## Pipeline changes

Same four-stage shape; one new data type flows through it.

### extract.py
- New dataclass: `Figure(caption: str, src: str, section: str | None)`.
- `Paper` gains `figures: list[Figure]`.
- Parse `<figure class="ltx_figure">` elements: `<img src>` +
  `<figcaption class="ltx_caption">`. Figures without a caption or without
  a raster `<img>` (SVG, multi-panel oddities) are skipped in v1.

### fetch.py
- Download figure images into the existing `~/.cache/paperdigest/` cache,
  keyed by arxiv id (e.g. `<cache>/<id>/figures/`).
- URL resolution gotcha (hit during the sanity check): `src` in arXiv HTML
  is relative — resolve against the final URL of the fetched HTML document
  (`https://arxiv.org/html/<id>vN/`), not against a guessed path.
- PNG/JPEG only in v1; skip others with a progress note.

### llm.py
- `Backend.complete()` gains an optional `images: list[bytes] | None`
  parameter (or a parallel `complete_with_images`; decide at implementation
  time — protocol change is the cleaner read).
- `OpenAICompatibleBackend`: encode as
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`
  content parts. Verified working against llama.cpp unchanged.
- `AnthropicBackend`: Anthropic's image content-block format.
- `FakeBackend` (tests/conftest): record that images were passed so the
  suite stays offline.

### digest.py
- New step after concept calls: one call per figure with image + caption +
  title + TLDR, per the prompt guidelines above.
- Progress line: `Explaining figure i/N: <caption first words>…`.
- Association: ask the outline step to map figures to concepts by section
  (it already assigns sections to concepts); unmatched figures go to the
  overview.
- `Digest` gains `figures: list[FigureNote(caption, body_md, image_path,
  concept_title | None)]`.

### render.py
- Copy each cached image into the paper folder (`fig1.png`, …).
- Embed in the matched concept note (or overview) as:

  ```markdown
  ## Figure: <caption>
  ![[fig1.png]]
  <explanation>
  ```

- Glossary and re-run guard semantics unchanged.

## Degradation & config

- New config key / flag: `figures = true` | `--figures/--no-figures`.
  Default **on**.
- If the backend rejects image content (non-vision model/server), catch the
  error on the *first* figure call, print one warning
  (`Warning: backend has no vision support; skipping N figures.`), and
  continue the digest without figures. Never fail the whole run over
  figures.
- Cost note for cloud users: images are token-expensive; README should say
  `--no-figures` is the budget option.

## Testing (offline, matching house style)

- Fixture HTML gains a `<figure>` block with a tiny checked-in PNG (a few
  hundred bytes).
- extract test: figure parsed with caption/src/section.
- digest test: FakeBackend asserts image bytes + caption + TLDR arrive in
  the figure call; figure count and association correct.
- render test: image file copied into folder, `![[…]]` embed present.
- cli test: `--no-figures` produces no figure calls; vision-rejection path
  warns and completes.
- No network, no real model, as always.

## Out of scope (v1 of this feature)

- SVG and multi-panel figure handling.
- Table extraction (tables are HTML, not images — different feature).
- OCR of text-dense figures / chart data extraction.
- Mermaid diagrams (separate idea: prompt-level change to concept notes,
  spec it independently if pursued).

## Open questions for implementation time

1. Protocol change vs. new method on `Backend` for images.
2. Whether to cap figures per paper (some papers have 20+; a `max_figures`
   config default ~8, ordered by document position, seems sane).
3. Whether the 9B model's chart-reading (loss curves, axes) is reliable
   enough to explain *results* figures, or whether v1 should prompt extra
   caution on plots vs. architecture diagrams. The sanity check only
   covered an architecture diagram.
