# Product audit: untouched directions

Date: 2026-07-13
Repository state audited: `ce55fef` (`master`)
Scope: current source, tests, README, design specifications, implementation plans, and Git history
Method: read-only source investigation plus three independent audit tracks covering runtime architecture, deferred work/history, and adversarial product ideation

> **⚠️ Superseded (2026-07-25).** Historical snapshot of `ce55fef` (2026-07-13);
> kept for provenance. Some findings are now out of date — notably, figure
> extraction and vision-based explanations have since shipped (see CHANGELOG),
> so any claim here that figures are unimplemented no longer holds. Line-number
> citations reflect the 2026-07-13 tree and have drifted from current source.

## Verdict

paperdigest currently implements two useful but disconnected one-way pipelines:

1. `digest`: arXiv paper → plain-English, wikilinked Obsidian notes.
2. `scaffold`: arXiv paper → Git-initialized research-code skeleton.

The strongest untouched territory is not another summary format. It is a shared, evidence-bearing representation that connects what a paper claims, where its evidence appears, and how the claim can be tested. That representation could make both existing pipelines more trustworthy and connect comprehension to reproduction.

## Current architecture

### Digest

The CLI fetches and extracts one arXiv paper, builds an outline, generates concept explanations and glossary definitions, then renders Markdown notes:

- CLI path: `paperdigest/cli.py:38-78`
- Paper extraction: `paperdigest/extract.py:33-71`
- Digest orchestration: `paperdigest/digest.py:125-199`
- Note rendering: `paperdigest/render.py:51-90`

The digest schema contains a TLDR, significance statement, concept notes, jargon, glossary definitions, and self-test questions. It does not retain claim-level evidence or exact source spans: `paperdigest/digest.py:20-30`.

### Scaffold

The scaffold pipeline asks an LLM to analyze the paper, plan modules, generate stubs and smoke tests, create a harness/configuration, and write project documentation:

- Analysis and planning prompts: `paperdigest/scaffold.py:38-101`
- Pipeline orchestration: `paperdigest/scaffold.py:178-269`
- Filesystem and Git materialization: `paperdigest/scaffold.py:272-317`

The result is intentionally a skeleton. Generated method bodies are normally TODO stubs, and generated smoke tests are instructed not to execute them: `paperdigest/scaffold.py:58-74`.

## Ranked untouched directions

### 1. Claim-to-evidence ledger

**Payoff:** very high
**Effort:** low to moderate
**Recommendation:** build first

Extract each important conclusion into a structured record containing:

- the exact claim
- exact supporting passage and source anchor
- equation, figure, or table identifier
- experimental conditions
- reported result
- whether the claim is stated or inferred
- assumptions and extraction uncertainty
- an observation that would falsify the claim
- a proposed reproduction experiment

Example:

```yaml
claim: Scaling depth improves validation accuracy
source:
  section: 4.2
  artifact: Table 3
  quote: "..."
status: stated
conditions:
  dataset: CIFAR-100
  architecture: ResNet
reported_result:
  baseline: 74.1
  proposed: 76.8
reproduction:
  config: configs/table3.yaml
  acceptance: accuracy >= 76.3
```

Current evidence gap:

- Digest outline fields stop at summary, concepts, jargon, and questions: `paperdigest/digest.py:45-55`.
- Scaffold analysis records components, datasets, hyperparameters, and experiment descriptions, but no claim/evidence relationship: `paperdigest/scaffold.py:38-46`.
- `Section` stores flattened text without paragraph, equation, table, or HTML-anchor identity: `paperdigest/extract.py:12-24`, `paperdigest/extract.py:50-61`.

This should become the shared intermediate artifact consumed by both `digest` and `scaffold`.

### 2. Reproduction feasibility and blocker report

**Payoff:** very high
**Effort:** low to moderate

Before creating a project, report whether the paper is realistically reproducible:

- required datasets and access status
- licenses or usage restrictions
- unavailable checkpoints or repositories
- stated and missing hyperparameters
- expected compute
- unclear preprocessing
- ambiguous evaluation procedures
- proprietary dependencies
- smallest meaningful reproduction target

The result should end with a blunt verdict: `ready`, `conditional`, or `blocked`.

The current analyzer extracts datasets, hyperparameters, and experiment names but does not assess availability, compute, ambiguity, or blockers: `paperdigest/scaffold.py:38-46`.

This prevents paperdigest from producing an impressive-looking repository for a paper that cannot be reproduced from the available information.

### 3. Structured result-table extraction

**Payoff:** high
**Effort:** moderate

Parse result and ablation tables into normalized records:

- dataset and split
- method and baseline
- metric and direction
- reported value and uncertainty
- experimental conditions
- exact table and cell provenance

Use those records to produce a “What the evidence actually shows” note and objective acceptance criteria for scaffold experiments.

Current evidence gap:

- Extraction flattens section content and preserves no table structure: `paperdigest/extract.py:50-61`.
- Scaffold experiments are prose descriptions of tables or figures: `paperdigest/scaffold.py:38-46`.
- Generated experiment documentation asks the model to restate reported numbers rather than consuming verified records: `paperdigest/scaffold.py:93-101`.

### 4. Paper-to-code traceability ledger

**Payoff:** high
**Effort:** moderate

Every generated API, function, configuration value, dependency, and TODO should identify its origin and status:

- `STATED`: explicitly specified by the paper
- `DERIVED`: directly derived from a stated equation or algorithm
- `IMPLEMENTATION CHOICE`: needed to build the software but not prescribed
- `UNKNOWN`: unresolved or unsupported

The present module planner invents a software API from paper-level components: `paperdigest/scaffold.py:48-56`. Stubs are asked to cite paper sections or equations, but those citations are coarse and do not prove the generated API or implementation choice came from the paper: `paperdigest/scaffold.py:58-66`.

A generated `TRACEABILITY.md` would expose plausible-looking inventions instead of hiding them inside valid Python.

### 5. Closed-loop reproduction audit

**Payoff:** high
**Effort:** moderate to high

Add an audit path that compares an evolving generated project against its paper contract:

- which components remain unimplemented
- which reported experiments have executable configs
- whether stated hyperparameters were preserved
- which reported tables or figures have corresponding runs
- whether measured results are genuinely comparable
- which implementation decisions lack provenance
- whether discrepancies indicate a code defect, missing paper detail, changed conditions, or insufficient compute

The scaffold currently stops after generation. Its harness is syntax-validated but not run against implemented method code: `paperdigest/scaffold.py:243-255`. Experiment tracking is append-only JSONL without source-grounded targets or discrepancy analysis.

The desired loop is:

```text
paper → claims → scaffold → runs → audit → evidence-backed verdict
```

### 6. arXiv revision delta

**Payoff:** medium to high
**Effort:** moderate

Compare two arXiv revisions and report changes to:

- claims
- reported metrics
- experiments and baselines
- equations
- caveats and limitations
- conclusions
- existing scaffold assumptions

Version suffixes are accepted by reference parsing, but there is no semantic revision model or comparison path. Re-running a digest either refuses or destructively replaces the existing paper folder: `paperdigest/render.py:44-55`.

Revision-aware output would protect hand-annotated notes and reveal when an existing reproduction project has become stale.

### 7. Static self-test answer contract

**Payoff:** moderate
**Effort:** low

Generate answer keys, source links, and misconception-specific hints for the existing self-test questions. The current output contains questions only: `paperdigest/digest.py:48-55`, `paperdigest/render.py:68-70`.

Keep this static. Turning it into an adaptive tutor or chat interface would add state and UI complexity without strengthening the core evidence pipeline.

### 8. Citation-derived prerequisite queue

**Payoff:** moderate
**Effort:** moderate

Extract only references needed to understand prerequisites, competing methods, datasets, or evaluation conventions. For each selected reference, explain why it should be read next.

References are currently discarded during extraction, and that behavior is tested: `paperdigest/extract.py:56-58`, `tests/test_extract.py:20-22`.

This should remain bounded to references already present in the paper. Expanding it into general paper discovery would create a separate product.

### 9. Cross-paper claim and contradiction map

**Payoff:** potentially very high
**Effort:** very high
**Sprawl risk:** high

Normalize claims, assumptions, datasets, metrics, and experimental conditions across papers. Highlight whether apparent disagreements are genuine contradictions or incomparable experiments.

The current vault-wide behavior is limited to filename-based glossary reuse:

- Existing terms are detected from filenames: `paperdigest/cli.py:68-72`.
- Existing glossary notes are never updated or attributed to additional papers: `paperdigest/render.py:80-88`.

Do not start here. This direction requires persistent canonical identities, metric normalization, indexing, corpus querying, and likely a new interaction surface. The claim ledger should prove useful on individual papers first.

## Existing backlog, not untouched ideas

### Figure explanations

Figure extraction and vision-based explanation are unimplemented, but not untouched. A detailed draft specification already exists and records a successful local multimodal feasibility test:

`docs/superpowers/specs/2026-07-11-figure-explanations.md`

Current source still has no figure data model, image extraction, multimodal backend path, or figure rendering:

- `Paper` has no figures: `paperdigest/extract.py:18-24`.
- Backend calls are text-only.
- `Digest` has no figure notes: `paperdigest/digest.py:20-30`.
- Renderer emits overview, concepts, and glossary only: `paperdigest/render.py:51-90`.

This remains the clearest implementation-ready backlog item, but it should not be presented as a newly discovered direction.

### Other deferred capabilities

The audit also confirmed these explicit or natural extensions:

- resumable/checkpointed digest generation
- PDF and non-arXiv input support
- SVG, multi-panel figure, OCR, and chart-data support
- adaptive model routing or escalation
- deeper automatic implementation of scaffold stubs
- MLflow/W&B integration

These improve coverage, reliability, or convenience. They do not address the larger scientific-trust gap as directly as claim provenance, feasibility assessment, and structured results.

## Recommended product boundary

Build a source-grounded intermediate artifact—provisionally `claims.json`—before adding more generation modes.

Minimum useful schema:

1. source anchors and verbatim evidence
2. claims and `stated`/`inferred` status
3. equations, figures, and tables linked to claims
4. experimental conditions and reported results
5. missing information and uncertainty
6. reproduction targets and acceptance criteria

Then:

- `digest` renders the artifact into understandable, auditable notes.
- `scaffold` converts reproduction targets into code structure and experiment configs.
- a future audit command compares real run records against those targets.

This is the smallest architectural move that turns paperdigest from a prose generator into trustworthy research infrastructure.

## Deferred direction recorded 2026-08-20: Paperdigest Quality Lab

This direction was proposed as a substantial future portfolio and product project. Thad likes the idea but explicitly chose not to pursue it on 2026-08-20. It is deferred, not rejected and not active work.

The proposed `paperdigest` v0.5.0-scale effort would add a source-grounded quality evaluation system around the existing product:

- a fixed corpus of real AI/ML papers and source-anchored evaluation cases;
- deterministic checks for broken links, invalid diagrams, missing sections, equation preservation, and source attribution;
- source-grounded assessment of unsupported claims, concept coverage, and explanation faithfulness;
- human review for uncertain findings;
- reproducible local-model comparisons using fixed configurations and repeated runs;
- an HTML report showing quality by paper, model, metric, and failure, with evidence drill-down;
- regression gates that prevent later releases from quietly reducing output quality.

The value is not another summary mode. It would turn an existing local-AI application into a measured product with an inspectable quality contract, documented failure modes, and release evidence. Revisit only when Thad deliberately selects it; do not treat this note as authorization to plan, implement, or run it.

## Audit safety and verification notes

- The audit itself was read-only.
- No source or runtime state was changed during investigation.
- Absence claims were checked against current source and documentation rather than inferred from README omissions alone.
- Three independent audit tracks converged on claim provenance and reproduction feasibility as the best first directions.
