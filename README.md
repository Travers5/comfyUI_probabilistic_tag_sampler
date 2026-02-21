# Probabilistic Tag Sampler (ComfyUI Custom Node)

A ComfyUI text node that samples tags from a probabilistic model made of base scores, a full influence matrix, and optional group filters/modifiers.

## Documentation index

- **Project intent**: [`INTENDED_BEHAVIOR.md`](INTENDED_BEHAVIOR.md)
- **Detailed code/program breakdown** (every program/sub-program + file IO): [`PROGRAM_STRUCTURE_AND_IO.md`](PROGRAM_STRUCTURE_AND_IO.md)
- **AI handoff/integration guide** (for other AI systems building around this node): [`AI_HANDOFF.md`](AI_HANDOFF.md)
- Change history notes: [`WHAT_HAS_BEEN_DONE.md`](WHAT_HAS_BEEN_DONE.md)

## Repository layout

- `__init__.py` – ComfyUI node registration/mappings.
- `probabilistic_tag_sampler.py` – node UI schema + tag sampling engine.
- `tag_model.py` – model schema, validation, JSON/CSV syncing, probability math.
- `LICENSE`
- `README.md`

Generated at runtime (next to `tag_model.py`):

- `tag_data.json` – canonical persisted model.
- `tag_data.csv` – spreadsheet-friendly editable matrix source.
- `prob_tag_sampler_debug.log` – optional debug output when enabled.

## Install

1. Place this repo under `ComfyUI/custom_nodes/comfyui_probabilistic_tag_sampler`.
2. Restart ComfyUI.
3. Add node **“Probabilistic Tag Sampler (Text)”** from category `text/tags`.

## Model editing workflow

- First run auto-creates `tag_data.json` + `tag_data.csv` from static defaults in `tag_model.py`.
- If CSV is newer than JSON, loader rebuilds JSON from CSV automatically.
- Edit either format, but CSV is better for matrix-heavy workflows.

## Quick behaviour summary

- Maps per-tag score `[-1..1]` to probability `[0..1]` using logistic mapping with hard 0%/100% at exact `-1/+1`.
- Selects tags iteratively by comparing probability vs RNG roll margin.
- Applies chosen-tag influence row after each selection to update remaining scores.
- Supports min/max tag constraints, input prompt chaining (`input_text`), optional weighted output, and dynamic group controls (Exclude/Boost/Suppress).

For complete technical details and file-level IO contracts, use:

- [`PROGRAM_STRUCTURE_AND_IO.md`](PROGRAM_STRUCTURE_AND_IO.md)
- [`AI_HANDOFF.md`](AI_HANDOFF.md)
