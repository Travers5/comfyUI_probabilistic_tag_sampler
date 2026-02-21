# Program Structure and Input/Output Detail

## 1) Program inventory

### `__init__.py`

Purpose:
- Registers the node class with ComfyUI.

Inputs:
- None directly.

Outputs/side effects:
- Exposes `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` so ComfyUI can discover the node.

---

### `tag_model.py`

Purpose:
- Defines model schema and defaults.
- Handles model persistence (`tag_data.json` / `tag_data.csv`).
- Performs schema validation.
- Provides probability and score-combination math.

Main sub-programs/functions:
- `TagModel` dataclass: in-memory model container.
- `score_to_probability(score)`: maps score to probability.
- `combine_scores(current, delta)`: merges existing score with influence update.
- `write_csv_template(csv_path, template)`: writes matrix template.
- `read_template_from_csv(csv_path)`: parses matrix CSV into schema dict.
- `load_tag_model()`: auto-generate/sync/load model with mtime cache.

Input files:
- `tag_data.csv` (optional, if exists and newer than JSON).
- `tag_data.json` (canonical persisted model).

Output files:
- `tag_data.json` (created/rewritten as needed).
- `tag_data.csv` (created on first run if missing).

Folder assumptions:
- Reads/writes files in same folder as `tag_model.py`.

---

### `probabilistic_tag_sampler.py`

Purpose:
- Implements the ComfyUI node UI contract and sampling algorithm.

Main sub-programs/functions:
- `ProbabilisticTagSampler.INPUT_TYPES()`: builds UI input schema including dynamic group toggles.
- `ProbabilisticTagSampler.generate(...)`: end-to-end generation pipeline.
- `_find_preselected_tags(...)`: parses known tags from `input_text`.
- `_apply_tag_influence_update(...)`: applies one selected tag's influence row.
- `_normalise_weights_mean_constrained(...)`: optional post-selection weight normalisation.
- Helper methods for RNG shaping, weight shaping, influence scaling, parsing, logging.

Runtime inputs (ComfyUI node fields):
- Required scalar inputs: `seed`, `input_text`, `min_tags`, `max_tags`, `separator`, `boost_amount`, `suppress_amount`, `output_strengths`, `rng_avg_count`, `normalise_strengths`.
- Required dynamic group booleans: `<group_name> (Exclude|Boost|Suppress)` for each model group.

Runtime outputs:
- One `STRING` output called `tags`.

Optional side-effect files:
- `prob_tag_sampler_debug.log` when debug logging is enabled.

---

## 2) Data files and exact schemas

### `tag_data.json`

Object fields:
- `group_names: string[]`
- `tags: string[]` length `N`
- `tag_group_masks: int[]` length `N`
- `base_scores: number[]` length `N`, each in `[-1,1]`
- `influence: number[][]` shape `N x N`, each in `[-1,1]`

### `tag_data.csv`

Format:
1. Optional comment metadata line:
   - `#group_names=name1|name2|...`
2. Header row:
   - `tag,base_score,group_mask,<tag1>,...,<tagN>`
3. `N` data rows:
   - `<tag_i>,<base_score_i>,<group_mask_i>,<influence_i_to_tag1>,...,<influence_i_to_tagN>`

Hard constraint:
- Header tag columns must exactly match row tag names in the same order.

---

## 3) Program-to-program integration flow

1. ComfyUI imports package -> `__init__.py` exposes class mapping.
2. Node class in `probabilistic_tag_sampler.py` calls `load_tag_model()` from `tag_model.py`.
3. `tag_model.py` ensures JSON/CSV presence and sync policy.
4. Node runs sampler using loaded `TagModel`.
5. Node emits final text string to downstream ComfyUI nodes.

This means all programs in repo are tightly integrated and required for full operation.
