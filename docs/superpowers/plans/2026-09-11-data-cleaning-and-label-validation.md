# Bio-Know-Tag Data Cleaning and Label Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local pipeline that exports the 458-label taxonomy, cleans and aggregates raw biology questions, and runs the first two DeepSeek label-understanding validation stages with resumable evidence and reports.

**Architecture:** Keep pure, testable transformations in `src/bio_know_tag/`; keep thin command-line entry points in `scripts/`. Inputs remain immutable under `source/raw/` or external paths, generated experiment artifacts go under timestamped `runtime/` directories, and curated taxonomy outputs live in `configs/`.

**Tech Stack:** Python 3.10+, standard library, BeautifulSoup4, openpyxl, pytest, OpenAI-compatible `/v1/chat/completions` over `urllib`.

## Global Constraints

- The canonical taxonomy input is `/Users/wishfine/Downloads/高中_生物_图谱_2026-08-20 09_42_27.xlsx` with exactly 458 non-empty labels.
- The raw question input on server is `/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/生物.jsonl`.
- DS endpoints are `http://172.22.0.35:9092/v1/chat/completions` and `http://172.22.0.35:9093/v1/chat/completions`; model is `DeepSeek-V4-Flash`.
- Every full run writes to `runtime/<timestamp>/`, emits JSONL evidence plus `report.json`, and supports restart without duplicating completed IDs.
- Question labeling must use the smallest sufficient knowledge-point set; background terms, tools, and distractors are not labels.
- No artificial gold labels are generated for Stage 3; human review is required before evaluating question-label accuracy.

---

### Task 1: Project scaffold and taxonomy export

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/bio_know_tag/__init__.py`
- Create: `src/bio_know_tag/labels.py`
- Create: `scripts/export_labels.py`
- Create: `tests/test_labels.py`
- Generate: `configs/labels.jsonl`
- Generate: `configs/labels.report.json`

**Interfaces:**
- Consumes: an `.xlsx` workbook containing the eight observed Chinese columns.
- Produces: `load_label_workbook(path: Path) -> list[dict[str, str]]` and `write_label_outputs(records, output_path, report_path) -> dict`.

- [ ] **Step 1: Write the failing taxonomy tests**

```python
def test_load_label_workbook_normalizes_nbsp_and_preserves_id(tmp_path):
    workbook = make_workbook(tmp_path, label_id="001", label_name="观察实验（旧）\u00a0")
    labels = load_label_workbook(workbook)
    assert labels[0]["label_id"] == "001"
    assert labels[0]["label_name"] == "观察实验（旧）"

def test_load_label_workbook_rejects_duplicate_names(tmp_path):
    workbook = make_workbook(tmp_path, names=["体液免疫", "体液免疫"])
    with pytest.raises(ValueError, match="duplicate label_name"):
        load_label_workbook(workbook)
```

- [ ] **Step 2: Run `pytest tests/test_labels.py -q` and confirm failure because `bio_know_tag.labels` does not exist**
- [ ] **Step 3: Implement exact column mapping, whitespace normalization, required-field and uniqueness validation**
- [ ] **Step 4: Run `pytest tests/test_labels.py -q` and confirm all taxonomy tests pass**
- [ ] **Step 5: Export the supplied workbook and verify 458 records, zero blanks, and zero duplicates**

### Task 2: Question cleaning and parent-child aggregation

**Files:**
- Create: `src/bio_know_tag/questions.py`
- Create: `scripts/preprocess_questions.py`
- Create: `tests/test_questions.py`

**Interfaces:**
- Consumes: JSONL rows with `question_id`, `parent_id`, string-or-dict `question_info`, and optional `parent_question_info`.
- Produces: `normalize_text(value) -> str`, `clean_question_info(value) -> dict`, and `aggregate_questions(rows) -> tuple[list[dict], dict]`.

- [ ] **Step 1: Write failing tests for HTML removal, `<sup>/<sub>` conversion, option formatting, child-before-parent input, standalone questions, malformed JSON, and duplicate question IDs**

```python
def test_aggregate_supports_child_before_parent():
    parents, report = aggregate_questions([child_row(), parent_row()])
    assert parents[0]["question_id"] == "p1"
    assert [q["question_id"] for q in parents[0]["sub_questions"]] == ["c1"]
    assert report["error"] == 0

def test_normalize_text_preserves_scientific_super_and_subscripts():
    assert normalize_text("CO<sub>2</sub> 与 10<sup>3</sup>") == "CO_{2} 与 10^{3}"
```

- [ ] **Step 2: Run `pytest tests/test_questions.py -q` and confirm expected missing-module failures**
- [ ] **Step 3: Implement streaming JSONL parsing and deterministic global aggregation without embedding file paths**
- [ ] **Step 4: Write output JSONL atomically and emit `report.json` containing input, processed, parent_count, child_count, skipped, and error**
- [ ] **Step 5: Run question tests and the complete test suite**

### Task 3: Resumable DS client and Stage 1 self-explanation

**Files:**
- Create: `src/bio_know_tag/ds.py`
- Create: `scripts/run_label_self_explain.py`
- Create: `tests/test_ds.py`

**Interfaces:**
- Consumes: `configs/labels.jsonl`, an endpoint, model name, and output directory.
- Produces: one evidence row per label with prompt, raw response, parsed response, endpoint, model, attempt count, latency, and error.

- [ ] **Step 1: Write failing tests for JSON extraction from plain/fenced responses, retryable HTTP errors, resume ID loading, and summary counts**

```python
def test_parse_json_content_accepts_markdown_fence():
    assert parse_json_content('```json\n{"core_meaning":"复制"}\n```')["core_meaning"] == "复制"

def test_completed_ids_only_includes_successful_rows(tmp_path):
    evidence = write_evidence(tmp_path, ok_id="1", error_id="2")
    assert load_completed_ids(evidence) == {"1"}
```

- [ ] **Step 2: Run `pytest tests/test_ds.py -q` and confirm expected failures**
- [ ] **Step 3: Implement the OpenAI-compatible client with timeout, bounded retries, endpoint round-robin, and strict response validation**
- [ ] **Step 4: Implement Stage 1 prompt and JSON schema: `core_meaning`, `included_content`, `excluded_content`**
- [ ] **Step 5: Emit evidence incrementally with fsync, skip successful IDs on resume, and atomically refresh `report.json` after every label**
- [ ] **Step 6: Run unit tests using a local fake HTTP server; do not require DS for the test suite**

### Task 4: Stage 2 alignment judge and disputed-label audit

**Files:**
- Create: `scripts/judge_label_alignment.py`
- Extend: `src/bio_know_tag/ds.py`
- Extend: `tests/test_ds.py`

**Interfaces:**
- Consumes: taxonomy rows and successful Stage 1 evidence.
- Produces: a judge evidence row with score 1–5, omissions, expansions, boundary differences, taxonomy audit decision, and an L1/L2/L3 recommendation.

- [ ] **Step 1: Write a failing validation test rejecting scores outside 1–5 and unknown audit decisions**
- [ ] **Step 2: Run the focused test and confirm it fails for the missing validator**
- [ ] **Step 3: Implement a single structured Judge prompt containing label name, original taxonomy fields, and Stage 1 explanation**
- [ ] **Step 4: Validate enums and map results to L1 (`score >= 4` and no taxonomy issue), L2 (`score >= 4` but boundaries required), or L3 (`score <= 3` or taxonomy issue)**
- [ ] **Step 5: Emit a separate `manual_review.jsonl` for all score ≤3/L3 items plus a reproducible high-score sample**
- [ ] **Step 6: Run focused and complete tests**

### Task 5: Operations documentation and smoke-test handoff

**Files:**
- Create: `README.md`
- Create: `docs/experiment-runbook.md`
- Create: `source/raw/README.md`
- Create: `runtime/.gitkeep`

**Interfaces:**
- Consumes: the tested scripts from Tasks 1–4.
- Produces: exact local commands and exact server commands for smoke and full runs.

- [ ] **Step 1: Document environment setup, taxonomy export, question preprocessing, DS connectivity, 3-label smoke, and 458-label full run**
- [ ] **Step 2: Document server synchronization under `/local_data/zhangyonglin/Bio-Know-Tag` and sibling data directory `/local_data/zhangyonglin/data`**
- [ ] **Step 3: Document `nohup`, PID capture, log/report/evidence monitoring, and acceptance checks `processed == input` and `error == 0`**
- [ ] **Step 4: Run `python -m compileall src scripts` and `pytest -q`**
- [ ] **Step 5: Inspect `git diff --check`, `git status`, and generated label report before committing**

### Task 6: Commit, push, server sync, and execution

**Files:**
- No new source files unless verification exposes a defect; defects require a failing regression test first.

**Interfaces:**
- Consumes: verified local repository and working SSH access to `xdf-35`.
- Produces: pushed `origin/main`, synchronized server checkout, timestamped smoke/full runtime directories, PID file, evidence, and final reports.

- [ ] **Step 1: Commit only reviewed project files and push `origin main`**
- [ ] **Step 2: On server, clone or fast-forward `/local_data/zhangyonglin/Bio-Know-Tag` and verify `git log -1 --oneline`**
- [ ] **Step 3: Create `/local_data/zhangyonglin/data`, reference the immutable raw JSONL, and run preprocessing smoke on a bounded sample**
- [ ] **Step 4: Verify DS with the minimal curl request, then run a 3-label Stage 1/2 smoke test**
- [ ] **Step 5: Start the full run with `nohup`, save PID immediately, and verify the process and first evidence records**
- [ ] **Step 6: Accept only when processed equals input, error is zero, evidence line counts match, and manual samples pass review**

