# Intended Behavior

This project provides a ComfyUI custom text node that generates prompt tags through **probabilistic sampling with dynamic dependencies**.

## Intended outcome

Given a seed and a tag model:

- Return a tag string (plain tags or weighted tags),
- Respect user constraints (`min_tags`, `max_tags`, separator, groups),
- React to already-present tags from `input_text` (chaining),
- Keep results reproducible for identical seed + model + settings.

## What makes this node different

Unlike simple random tag pickers, this node uses:

1. **Per-tag base score** (`base_scores`) – initial tendency for each tag.
2. **Influence matrix** (`influence`) – selecting tag A nudges scores of all other tags.
3. **Group controls** (`Exclude`, `Boost`, `Suppress`) – runtime scenario filtering.
4. **Selection loop with probability-vs-roll margin** – prioritises strongest acceptance events.

## Non-goals

- This repo does not include model training.
- This repo does not include GUI dashboards for matrix authoring (CSV/edit externally).
- This repo does not guarantee semantic quality of tags; quality depends on user-supplied model data.
