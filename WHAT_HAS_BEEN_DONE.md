# What has been done (development notes)

This document summarises the major features and changes implemented in this project so far, and why they were added.

It is written as a practical “history + rationale” note for GitHub, so future changes are easier to track.

---

## Current files and their roles

- `__init__.py`
  - Registers the node class with ComfyUI via:
    - `NODE_CLASS_MAPPINGS`
    - `NODE_DISPLAY_NAME_MAPPINGS`

- `tag_model.py`
  - Defines the tag model schema and loading pipeline.
  - Holds a static Python “template” (constants) that can be edited to define your initial tags/groups/matrix.
  - Auto-generates:
    - `tag_data.json` (machine-readable)
    - `tag_data.csv` (human-editable matrix)
  - Rebuilds JSON from CSV when CSV is newer than JSON.
  - Provides model math:
    - `score_to_probability()`: logistic mapping from `[-1,1]` to `[0,1]`
    - `combine_scores()`: score update combiner with hard lock precedence and disagreement-weighted blending

- `probabilistic_tag_sampler.py`
  - Implements the ComfyUI node logic.
  - Performs probabilistic selection under min/max constraints.
  - Applies group-based Exclude/Boost/Suppress controls.
  - Supports strength output + strength normalisation.
  - Supports chaining from existing prompts via `input_text`.
  - Includes code-only debugging log settings.

---

## Major implemented features

### 1) ComfyUI node framework

- A clean, minimal ComfyUI custom node structure:
  - `INPUT_TYPES`
  - `RETURN_TYPES`
  - `FUNCTION`
  - `CATEGORY`
- Registered through `__init__.py` so ComfyUI discovers the node.

Why:
- Establish a stable base for ongoing expansion without fighting ComfyUI’s expectations.

---

### 2) Tag model stored as static template + generated JSON/CSV

- `tag_model.py` contains static constants:
  - group names
  - tag names
  - group bitmasks
  - base scores
  - NxN influence matrix
- On first run, the system auto-generates:
  - `tag_data.json`
  - `tag_data.csv`
- If CSV is edited and becomes newer than JSON, the loader rebuilds JSON from CSV.

Why:
- You wanted the “starting values” to be editable directly inside Python for initial bootstrapping.
- You also wanted an easy spreadsheet workflow for managing the influence matrix.

Key properties:
- The CSV must remain a square matrix.
- Column tag headers must match the row tag names in the same order.

---

### 3) Probability mapping from score

- Score range: `[-1, 1]`
- Probability range: `[0, 1]`
- Mapping: logistic “S curve”
- Hard clamps:
  - score `<= -1` maps to exactly `0%`
  - score `>= +1` maps to exactly `100%`

Why:
- This preserves the intuitive “-1 is never, +1 is always” semantics.
- Intermediate scores stay smooth and usable as “likelihood” values.

---

### 4) Influence updates + improved score combiner

After each chosen tag:

- The node reads one row of the influence matrix:
  - `delta = influence[chosen][other]`
- It updates each remaining tag using `combine_scores(current, delta)`.

The combiner includes:

- Hard lock precedence:
  - `-1` always wins
  - `+1` wins unless a `-1` is involved
- Optional “reinforcement snap”:
  - If `current` is already near certainty and `delta` reinforces strongly, snap to `+1` or `-1`
- Otherwise:
  - Blend in atanh-space using a disagreement-dependent weight:
    - small change when new value is similar
    - larger change when new value conflicts

Why:
- You observed scores were saturating too quickly toward extremes.
- You wanted new information to have more effect when it conflicts with the current belief,
  but minimal effect when it merely repeats what you already believe.

---

### 5) Selection loop logic with min/max constraints

Per iteration:

- For every candidate tag:
  - compute `p = score_to_probability(score)`
  - roll random `r`
  - margin = `p - r`
- Choose the tag with the largest positive margin.
- If none are positive:
  - stop if `min_tags` satisfied
  - otherwise “force” the best negative margin (closest to acceptance)

After the main loop:

- If still below `min_tags`, fill greedily with the highest-probability remaining tags.

Why:
- This matches your original concept:
  - “pick what most strongly exceeded its roll”
  - but still guarantee `min_tags` if required.

---

### 6) RNG shaping by averaging multiple random draws

Instead of one `rng.random()`:

- Draw `K` uniform randoms
- Use their average

Effects:

- K=1 -> uniform
- K=2 -> triangular-ish around 0.5
- larger K -> tighter around 0.5

Why:
- You wanted to reduce extremely high/low random values and favour mid-range rolls.

---

### 7) Optional strength output, influence scaling, and normalisation

When “strengths” are enabled:

- Each selected tag gets a weight.
- Weight is derived from its margin (with separate shaping for forced selections).
- Influence deltas can be scaled by a factor derived from the weight.

Then optionally:

- Normalise selected weights so the average is ~1.0,
  while respecting minimum/maximum weight constraints.

Why:
- You wanted a wider spread of weights (not all near the top end).
- You wanted an option to force the mean weight to 1.0 but without automatically pushing everything to extremes.

---

### 8) Group controls (Exclude / Boost / Suppress)

For each group:

- Exclude:
  - hard lock to `-1.0` (0% probability)
- Boost:
  - add `+boost_amount` to the current score
- Suppress:
  - add `-suppress_amount` to the current score

Groups are defined by per-tag bitmasks, so tags can belong to multiple groups.

Why:
- You needed an easy way to disable whole sets of tags or shift their likelihood without editing the matrix.

---

### 9) Chaining / continuation via input_text (major change)

New input:

- `input_text` (multiline string)

Behaviour:

- Extract tags from `input_text` that match the known tag list.
- Treat them as already selected.
- Apply their influence updates in order.
- Continue sampling further tags from that updated state.

Extraction strategies:

1. Explicit weighted patterns:
   - `(tag:1.23)`
2. Delimited token lists (comma/newline/; /|), optionally with `:weight`
3. Fallback “word-ish boundary” scan in free-form text

Duplicates are ignored (first occurrence wins). Excluded tags are ignored.

Why:
- You wanted to chain multiple sampler nodes, and also allow other systems to provide partial tags/prompts
  which this node can recognise and extend.

---

### 10) Seed control duplication clean-up

ComfyUI frequently treats an input named `seed` specially and may display its own “control after generate” UI.

To avoid duplicated and confusing controls:

- The node defines only `seed`.
- It does not define a separate `control_after_generate` field.

Why:
- You observed two similar controls in the UI, and only one seemed effective.
- The clean approach is to rely on ComfyUI’s built-in behaviour for seed inputs.

---

## Known follow-ups / next likely improvements

- Add explicit options for how strictly to match tags in `input_text` (for example:
  “only parse delimited lists” vs “also scan free-form text”).
- Add UI toggles for code-only constants (debug logging and preselected weight behaviour),
  if you decide you want them user-facing later.
- Expand the static template from 6 tags to your full 59+ tags once your final tag set is decided.
