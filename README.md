# Probabilistic Tag Sampler (ComfyUI Custom Node)

A ComfyUI custom node that generates a tag string by sampling from a set of tags using:

- A per-tag *base score* in `[-1, 1]` (mapped to a probability via an S-curve)
- A full `N x N` *influence matrix* in `[-1, 1]` that updates the remaining tags after each selection
- Optional group controls (Exclude / Boost / Suppress) driven by per-tag group bitmasks
- Optional output “strengths” (weights) for each selected tag, and optional strength normalisation
- Optional chaining: you can feed an existing text prompt into the node, it will extract tags it recognises and continue sampling from that state

This repository is intended to be editable: you can expand from a small starter template (6 tags) to your full list (59+ tags), and maintain the model either by editing a JSON file or by editing a generated CSV file.

---

## Repository / folder layout

Typical layout inside your ComfyUI custom nodes folder:

```
ComfyUI/
  custom_nodes/
    comfyui_probabilistic_tag_sampler/
      __init__.py
      probabilistic_tag_sampler.py
      tag_model.py
      tag_data.json   (auto-generated on first run)
      tag_data.csv    (auto-generated on first run)
```

Notes:

- `__init__.py` registers the node with ComfyUI.
- `probabilistic_tag_sampler.py` is the node implementation.
- `tag_model.py` loads the tag model and manages `tag_data.json` / `tag_data.csv`.
- `tag_data.json` and `tag_data.csv` are created next to `tag_model.py` the first time ComfyUI imports the node (or the first time the node runs).

---

## Installation

1. Copy this repository folder into:

   `ComfyUI/custom_nodes/`

2. Ensure the main node file is named:

   `probabilistic_tag_sampler.py`

   (If you keep versioned filenames, make sure `__init__.py` imports the correct module name.)

3. Restart ComfyUI.

4. Add the node in the graph:

   The display name is:

   `Probabilistic Tag Sampler (Text)`

   Category:

   `text/tags`

---

## First run behaviour

On first load/run, the tag model loader will:

1. Use the static Python constants in `tag_model.py` as a default template.
2. Create `tag_data.json` if it does not exist.
3. Create `tag_data.csv` if it does not exist.
4. If `tag_data.csv` exists and is *newer* than `tag_data.json`, it rebuilds `tag_data.json` from the CSV.
5. Loads `tag_data.json` (with a simple mtime cache).

This means you can either:

- Edit `tag_data.json` directly, or
- Edit `tag_data.csv` (recommended for large matrices), then save it; the loader will rebuild `tag_data.json` automatically.

---

## Tag model data

### Concepts

- **Score**: a float in `[-1, 1]`.
  - `-1` means “hard lock to 0%”.
  - `+1` means “hard lock to 100%”.
  - `0` means “50%”.
- **Probability mapping**: score -> probability is an S-curve (logistic).
- **Influence**: `influence[a][b]` is the *delta score* applied to tag `b` when tag `a` is selected.
- **Groups**: each tag can belong to zero or more “groups” via a bitmask:
  - Group 1 => bit 0 (value `1`)
  - Group 2 => bit 1 (value `2`)
  - Group 3 => bit 2 (value `4`)
  - and so on

Groups drive the UI controls for Exclude / Boost / Suppress.

### JSON schema (`tag_data.json`)

`tag_data.json` is a plain JSON object with:

- `group_names`: list of group display names
- `tags`: list of tag strings (length `N`)
- `tag_group_masks`: list of integers (length `N`)
- `base_scores`: list of floats in `[-1, 1]` (length `N`)
- `influence`: 2D list of floats in `[-1, 1]` (shape `N x N`)

### CSV schema (`tag_data.csv`)

The CSV is generated as a single square matrix file:

- First line: group names, in a comment:
  - `#group_names=Group 01|Group 02|...`
- Header row:
  - `tag,base_score,group_mask,<tag1>,<tag2>,...,<tagN>`
- Then `N` rows, one row per tag:
  - `tag_name, base_score, group_mask, influence_row_values...`

Important rule:

- The tag column headers **must exactly match** the tag names down the first column, in the same order (so the matrix is unambiguous).

---

## Node inputs

### Core controls

- **seed** (INT)  
  ComfyUI treats an input named `seed` specially and may show a built-in “control after generate” option.  
  This node intentionally does *not* define a separate `control_after_generate` input to avoid duplicated seed controls.

- **input_text** (STRING, multiline)  
  Optional. The node will try to extract known tags from this text and treat them as already selected.

- **min_tags** (INT)  
  Minimum number of tags to output.

- **max_tags** (INT)  
  Maximum number of tags to output. Use `-1` for “no limit” (internally limited to remaining non-excluded tags).

- **separator**  
  How tags are joined in the output string:
  - `comma_space`, `comma`, `space`, `newline`

### Group controls

For each group name defined in `tag_data.json` / `tag_data.csv`, the UI will show:

- `<Group Name> (Exclude)` – sets tags in that group to a hard 0% (`-1`)
- `<Group Name> (Boost)` – adds `+boost_amount` to their current score
- `<Group Name> (Suppress)` – adds `-suppress_amount` to their current score

Controls can be combined; Exclude is applied as a hard lock.

### Strength controls (optional)

- **output_strengths** (BOOLEAN)
  - Off: output is plain tags: `tag_a, tag_b, tag_c`
  - On: output uses weighted prompt syntax: `(tag_a:1.10), (tag_b:0.45), ...`

- **normalise_strengths** (BOOLEAN)  
  If strengths are on, you can normalise strengths after selection so the *average* is 1.0, while respecting min/max limits.

### Random shaping

- **rng_avg_count** (INT)
  Instead of using a single uniform random number `r`, the node uses the average of `K` uniform draws:
  - `K=1`: uniform
  - `K=2`: triangular-ish around 0.5
  - larger `K`: increasingly concentrated around 0.5

This lets you tune how often borderline tags are accepted/rejected.

---

## Output format

- Output type: `STRING`
- Output name: `tags`

When `output_strengths` is enabled, weights are clamped to:

- min: `0.10`
- max: `2.00`

Interpretation (by design in this project):

- Around `0.50` means the tag only just met the threshold.
- Below `0.50` means the tag was forced in to satisfy `min_tags`.
- Higher values indicate higher priority.

---

## Chaining / continuation via input_text

If you pass an existing prompt into **input_text**, the node:

1. Extracts any known tags it can find.
2. Treats them as already selected (up to `max_tags`).
3. Applies their influence updates in order.
4. Continues sampling additional tags from the updated state.

This enables patterns like:

- Node A produces a set of tags, Node B continues from them.
- Another system generates a prompt containing some tags; this node picks out the tags it recognises and extends them.

### Tag extraction rules

The node attempts extraction in this order:

1. Weighted patterns anywhere:
   - `(tag:1.23)`
2. Delimited token lists:
   - `tag1, tag2, tag3`
   - `tag1:1.2, tag2:0.7`
   - delimiters: comma, newline, `;`, `|`
3. Fallback scan for tag names in free-form text using “word-ish” boundaries.

Duplicates are ignored (first occurrence wins). Tags that are excluded by group settings are ignored even if found in the text.

### Using weights found in input_text

If the prompt contains explicit weights like `(tag:1.20)`, the node can optionally use those weights to scale influence updates for the preselected tags (this is enabled by default in code).

---

## Debugging and logs

There are code-only toggles near the top of `probabilistic_tag_sampler.py`:

- `DEBUG_LOG_TO_FILE`
- `DEBUG_LOG_APPEND`
- `DEBUG_LOG_FILENAME`
- `DEBUG_LOG_INCLUDE_POST_UPDATE_PROBS`
- `PRINT_SUMMARY_TO_CONSOLE`

When enabled, a log file is written next to `probabilistic_tag_sampler.py`, containing:

- Per-iteration candidate probabilities and RNG samples
- The chosen tag each iteration
- Optional post-update probabilities
- Final output

---

## Customising for your real tag set (59+ tags)

You typically adjust three places:

1. **`tag_model.py` static constants**  
   - `DEFAULT_GROUP_NAMES`
   - `DEFAULT_TAGS`
   - `DEFAULT_TAG_GROUP_MASKS`
   - `DEFAULT_BASE_SCORES`
   - `DEFAULT_INFLUENCE`

   These constants are the “source of truth” for the initial auto-generated files.

2. **`tag_data.csv`** (recommended for large edits)  
   - Edit in a spreadsheet application.
   - Ensure the matrix stays square and the tag headers match the first column.

3. **`tag_data.json`** (direct edits)  
   - Works fine but is harder for large matrices.

Recommended workflow:

- Expand the constants in `tag_model.py` to your full size once.
- Let the node auto-generate `tag_data.csv`.
- Maintain the matrix by editing `tag_data.csv` after that.

---

## Known limitations

- If your tags are common English words (for example “cat”), the fallback scan can match them inside a longer free-form prompt. If this causes issues, prefer using a delimited tag list or the weighted `(tag:1.23)` format.
- The node assumes the directory is writable so it can create `tag_data.json`, `tag_data.csv`, and (optionally) debug logs.

---

## Licence
MIT
