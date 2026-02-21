# AI Handoff and Integration Contract

This file is written for other AI systems that need to extend this project or integrate around it.

## What this program does

- Implements a ComfyUI custom node that samples tags from a mutable probabilistic model.
- Produces either plain or weighted prompt strings.
- Can continue from partial prompts by extracting known tags from `input_text`.

## Inputs expected by the program

### A) Static/source inputs
- Python source files in this repo.
- Runtime model files in node directory:
  - `tag_data.json` and/or `tag_data.csv`.

### B) Runtime node invocation inputs
- `seed`: RNG seed for determinism.
- `input_text`: existing prompt text to parse for known tags.
- `min_tags` / `max_tags`: output count bounds.
- `separator`: output join mode.
- Group toggles: `<group> (Exclude|Boost|Suppress)`.
- `boost_amount` / `suppress_amount`.
- `output_strengths` / `normalise_strengths`.
- `rng_avg_count`.

## Outputs produced

### Primary output
- Single text string (`tags`) from ComfyUI node.

Possible formats:
- Plain: `tag_a, tag_b, tag_c`
- Weighted: `(tag_a:1.10), (tag_b:0.62)`

### Side-effect output files
- `tag_data.json` created/updated by sync logic.
- `tag_data.csv` auto-created if absent.
- `prob_tag_sampler_debug.log` if debug enabled.

## Integration guidance for other AIs

### If you want to generate inputs for this program
- Produce valid `tag_data.csv` or `tag_data.json` with exact schema constraints.
- Ensure matrix is square and tag ordering is consistent.
- Keep all scores/influences inside `[-1,1]`.

### If you want to consume outputs from this program
- Parse result as either comma/newline/space-separated tags or `(tag:weight)` syntax.
- Do not assume every run has weights; check `output_strengths` in workflow config.

### If you want to extend this program
- Add new node inputs in `INPUT_TYPES()` and plumb through `generate()`.
- Keep deterministic behavior for fixed seed/settings.
- Preserve `load_tag_model()` auto-sync contract unless intentionally versioning format.
