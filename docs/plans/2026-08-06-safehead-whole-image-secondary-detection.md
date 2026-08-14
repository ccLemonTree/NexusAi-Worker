# Safehead Whole-Image Secondary Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Return first-label Safehead detections only when they neither intersect nor lie near any second-label detection from a single whole-image inference.

**Architecture:** Keep the existing `Model.execute` entry point, but separate the box-proximity rule into a small pure helper. When first-label results exist, run the configured second label once against the full image, then filter each first-label box by testing it against every second-label box after expanding the first box by 20% of its width and height.

**Tech Stack:** Python, pytest, existing `BoundingBox` objects and `analyseRun` inference API.

---

## Understanding summary

- `self.logicResult` contains the first-label (`personnew`) detections.
- The second label is read from `cfg.logicModelDict[self.logicModelName][1]["label"]` (`yolov5nohead` in current configuration).
- Second-label inference runs only when at least one first-label result exists.
- Second-label inference receives the whole image and runs once per `execute` call.
- A second-label box is considered related when it intersects the first box or intersects that first box expanded by 20% on every side.
- Only first-label boxes unrelated to every second-label box are returned.
- Model configuration and other pipelines are outside this change.

## Assumptions

- “Expand by 20%” means adding 20% of the first box width to both left and right and 20% of its height to both top and bottom.
- Edge contact counts as intersection/nearby.
- If second-label inference returns no boxes, all first-label boxes remain eligible.
- The returned box keeps its original coordinates and only its `classname` is changed, matching current behavior.

## Decision log

- Use proportional expansion instead of a fixed pixel threshold so behavior scales with image and target size.
- Expand only the first-label box because it is the candidate being evaluated and provides a stable reference size.
- Use axis-aligned rectangle overlap rather than center distance to handle differently sized detections.
- Keep one whole-image inference outside the first-result loop to avoid duplicate inference work.

### Task 1: Geometry regression tests

**Files:**
- Create: `tests/test_safehead.py`
- Test: `tests/test_safehead.py`

**Step 1: Write failing tests**

Add focused tests for direct overlap, a gap inside the 20% margin, a gap outside the margin, edge contact, and an empty second-label list.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_safehead.py -v`

Expected: FAIL because the Safehead proximity/filter helper does not exist yet.

### Task 2: Whole-image filtering implementation

**Files:**
- Modify: `api/infer/Model_pipline/Safehead/Safehead.py`
- Test: `tests/test_safehead.py`

**Step 1: Add the minimal geometry helper**

Implement axis-aligned overlap against the first box expanded by a default ratio of `0.2`.

**Step 2: Update `Model.execute`**

Return early when no first-label results exist. Otherwise call `analyseRun` once with `[self.picture, self.picture]`, then retain only first-label boxes for which no second-label box overlaps the expanded area.

**Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_safehead.py -v`

Expected: PASS.

### Task 3: Repository-level verification

**Files:**
- Verify: `api/infer/Model_pipline/Safehead/Safehead.py`
- Verify: `tests/test_safehead.py`

**Step 1: Run syntax validation**

Run: `python -m compileall kafka api tools utils`

Expected: exit code 0.

**Step 2: Re-run focused tests**

Run: `python -m pytest tests/test_safehead.py -v`

Expected: all tests pass with no failures.
