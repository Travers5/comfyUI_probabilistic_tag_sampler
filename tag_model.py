# v0.04.00

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
from typing import List, Optional, Tuple


# -------------------------------------------------------------------------
# STATIC DEFAULT TEMPLATE (edit this later to your real starting values)
# -------------------------------------------------------------------------
# You asked for a static matrix inside Python so you can later replace the
# placeholders with your "correct" starting values and have auto-generation
# use those exact values.

DEFAULT_GROUP_NAMES: List[str] = [
    "Group 01", "Group 02", "Group 03", "Group 04", "Group 05",
    "Group 06", "Group 07", "Group 08", "Group 09", "Group 10",
]

# 6 temporary tags for now (you can expand to 59+ later)
DEFAULT_TAGS: List[str] = [
    "tag_01", "tag_02", "tag_03", "tag_04", "tag_05", "tag_06",
]

# Bitmask per tag: 0 = belongs to no groups.
# Group 1 => bit 0 (value 1), group 2 => bit 1 (value 2), group 3 => bit 2 (value 4), etc.
DEFAULT_TAG_GROUP_MASKS: List[int] = [
    0, 0, 0, 0, 0, 0
]

# Base scores in [-1, 1]. 0 means 50% probability after the S-curve.
DEFAULT_BASE_SCORES: List[float] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Influence matrix NxN in [-1, 1].
# influence[row_tag][col_tag] is the delta applied to col_tag when row_tag is chosen.
DEFAULT_INFLUENCE: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_01 to others
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_02
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_03
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_04
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_05
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # from tag_06
]


def _default_template_dict() -> dict:
    """
    Converts the static constants above into the on-disk JSON schema.
    """
    return {
        "group_names": list(DEFAULT_GROUP_NAMES),
        "tags": list(DEFAULT_TAGS),
        "tag_group_masks": list(DEFAULT_TAG_GROUP_MASKS),
        "base_scores": list(DEFAULT_BASE_SCORES),
        "influence": [list(row) for row in DEFAULT_INFLUENCE],
    }


# -------------------------------------------------------------------------
# MODEL + MATH HELPERS
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class TagModel:
    tag_names: List[str]               # length N
    base_scores: List[float]           # length N, each in [-1, 1]
    influence: List[List[float]]       # NxN, each in [-1, 1]
    group_names: List[str]             # length M
    tag_group_masks: List[int]         # length N, each >= 0 (bitmask)

    def size(self) -> int:
        return len(self.tag_names)

    def group_count(self) -> int:
        return len(self.group_names)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def score_to_probability(score: float, alpha: float = 4.0) -> float:
    """
    Maps score in [-1, 1] to probability in [0, 1] using an S-curve (logistic),
    with hard clamps at -1 and +1 to force exactly 0% / 100%.
    """
    if score <= -1.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-alpha * score))


# v0.06.00 (replace only this function; it uses math, so ensure tag_model.py imports math)

def combine_scores(
    current: float,
    delta: float,
    *,
    base_weight: float = 0.02,   # small default influence of new value
    gamma: float = 3.0,          # higher => even smaller updates when values are similar
    lock_current: float = 0.99,  # if current already this high/low...
    lock_reinforce: float = 0.90,# ...and delta reinforces above this, snap to +/-1
    eps: float = 1e-6,           # prevents atanh overflow near +/-1
) -> float:
    """
    Non-saturating combiner with lock precedence.

    Locks:
      -1 always wins.
      +1 wins unless -1 involved.
      Optional reinforcement snap to +/-1 if already near certainty and new value reinforces.

    Otherwise:
      Update in atanh-space with a disagreement-dependent mixing weight.
    """

    # Hard locks: -1 dominates everything
    if current <= -1.0 or delta <= -1.0:
        return -1.0

    # Hard locks: +1 dominates if no -1 involved
    if current >= 1.0 or delta >= 1.0:
        return 1.0

    # Reinforcement snap (prevents slowly creeping to 1 from moderate updates)
    if current >= lock_current and delta >= lock_reinforce:
        return 1.0
    if current <= -lock_current and delta <= -lock_reinforce:
        return -1.0

    # Disagreement-based weighting:
    # d in [0..1]
    d = abs(delta - current) * 0.5
    if d < 0.0:
        d = 0.0
    if d > 1.0:
        d = 1.0

    # Weight for "delta" (new value). Small when similar, larger when conflicting.
    w = float(base_weight) + (1.0 - float(base_weight)) * (d ** float(gamma))
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0

    # atanh-space blend
    c = max(-1.0 + eps, min(1.0 - eps, float(current)))
    u = max(-1.0 + eps, min(1.0 - eps, float(delta)))

    x_c = math.atanh(c)
    x_u = math.atanh(u)

    x_new = (1.0 - w) * x_c + w * x_u
    out = math.tanh(x_new)

    # Final safety clamp
    if out < -1.0:
        return -1.0
    if out > 1.0:
        return 1.0
    return out




def _validate_template_dict(d: dict) -> None:
    group_names = d.get("group_names")
    tags = d.get("tags")
    tag_group_masks = d.get("tag_group_masks")
    base_scores = d.get("base_scores")
    influence = d.get("influence")

    if not isinstance(group_names, list) or not all(isinstance(x, str) for x in group_names):
        raise ValueError("'group_names' must be a list of strings")
    if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
        raise ValueError("'tags' must be a list of strings")
    if not isinstance(tag_group_masks, list) or not all(isinstance(x, int) for x in tag_group_masks):
        raise ValueError("'tag_group_masks' must be a list of integers")
    if not isinstance(base_scores, list) or not all(isinstance(x, (int, float)) for x in base_scores):
        raise ValueError("'base_scores' must be a list of numbers")
    if not isinstance(influence, list) or not all(isinstance(r, list) for r in influence):
        raise ValueError("'influence' must be a 2D list")

    n = len(tags)
    if len(tag_group_masks) != n:
        raise ValueError("tag_group_masks length must match tags length")
    if len(base_scores) != n:
        raise ValueError("base_scores length must match tags length")
    if len(influence) != n:
        raise ValueError("influence row count must match tags length")
    for r, row in enumerate(influence):
        if len(row) != n:
            raise ValueError(f"influence row {r} length must match tags length")

    for i, s in enumerate(base_scores):
        if not (-1.0 <= float(s) <= 1.0):
            raise ValueError(f"base_scores[{i}] out of range [-1,1]")
    for r in range(n):
        for c in range(n):
            v = float(influence[r][c])
            if not (-1.0 <= v <= 1.0):
                raise ValueError(f"influence[{r}][{c}] out of range [-1,1]")
    for i, mask in enumerate(tag_group_masks):
        if mask < 0:
            raise ValueError(f"tag_group_masks[{i}] must be >= 0")


def _dict_to_model(d: dict) -> TagModel:
    _validate_template_dict(d)
    return TagModel(
        group_names=list(d["group_names"]),
        tag_names=list(d["tags"]),
        tag_group_masks=[int(x) for x in d["tag_group_masks"]],
        base_scores=[float(x) for x in d["base_scores"]],
        influence=[[float(x) for x in row] for row in d["influence"]],
    )


# -------------------------------------------------------------------------
# CSV TEMPLATE + CSV -> JSON
# -------------------------------------------------------------------------

CSV_GROUP_NAMES_PREFIX = "#group_names="


def write_csv_template(csv_path: str, template: dict) -> None:
    """
    Writes a single CSV file that contains:
      - A comment line with group_names
      - A header row: tag,base_score,group_mask,<tag1>,<tag2>,...,<tagN>
      - N rows where each row supplies one influence row
    """
    _validate_template_dict(template)

    group_names: List[str] = template["group_names"]
    tags: List[str] = template["tags"]
    base_scores: List[float] = template["base_scores"]
    masks: List[int] = template["tag_group_masks"]
    influence: List[List[float]] = template["influence"]

    header = ["tag", "base_score", "group_mask"] + tags

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(CSV_GROUP_NAMES_PREFIX + "|".join(group_names) + "\n")
        w = csv.writer(f)
        w.writerow(header)
        for i, tag in enumerate(tags):
            row = [tag, f"{float(base_scores[i]):.6f}", str(int(masks[i]))]
            row += [f"{float(x):.6f}" for x in influence[i]]
            w.writerow(row)


def read_template_from_csv(csv_path: str) -> dict:
    """
    Parses the CSV format produced by write_csv_template().
    Returns a template dict matching the JSON schema.
    """
    group_names: List[str] = []

    # Read raw lines first to pick up group_names comment
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        raw_lines = f.readlines()

    data_lines: List[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(CSV_GROUP_NAMES_PREFIX):
            group_names = s[len(CSV_GROUP_NAMES_PREFIX):].split("|") if s[len(CSV_GROUP_NAMES_PREFIX):] else []
            continue
        if s.startswith("#"):
            # Ignore other comment lines
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError("CSV contains no table data.")

    # Parse the remaining lines as CSV
    reader = csv.reader(data_lines)
    header = next(reader, None)
    if header is None:
        raise ValueError("CSV missing header row.")

    if len(header) < 4:
        raise ValueError("CSV header too short; expected tag,base_score,group_mask plus tag columns.")

    if header[0] != "tag" or header[1] != "base_score" or header[2] != "group_mask":
        raise ValueError("CSV header must begin with: tag,base_score,group_mask")

    tag_columns = header[3:]
    if len(tag_columns) == 0:
        raise ValueError("CSV must include at least one tag column after group_mask.")

    tags: List[str] = []
    base_scores: List[float] = []
    masks: List[int] = []
    influence: List[List[float]] = []

    for row in reader:
        if not row or all(not x.strip() for x in row):
            continue
        if len(row) != len(header):
            raise ValueError(f"CSV row length {len(row)} does not match header length {len(header)}")

        tag_name = row[0].strip()
        if not tag_name:
            raise ValueError("CSV row has empty tag name.")

        tags.append(tag_name)
        base_scores.append(float(row[1]))
        masks.append(int(row[2]))

        infl_values = [float(x) for x in row[3:]]
        influence.append(infl_values)

    n = len(tags)
    if n != len(tag_columns):
        raise ValueError(
            "CSV must be a square matrix: number of rows must match number of influence tag columns.\n"
            f"Rows={n}, Columns={len(tag_columns)}"
        )

    # Enforce that row tag names match the column tag names (same order)
    if tags != tag_columns:
        raise ValueError(
            "CSV tag order mismatch.\n"
            "The tag column headers (after group_mask) must exactly match the tag names down the first column,\n"
            "in the same order.\n"
            f"Row tags: {tags}\n"
            f"Col tags: {tag_columns}"
        )

    # If group_names not present in CSV, keep defaults from the Python template
    if not group_names:
        group_names = list(DEFAULT_GROUP_NAMES)

    template = {
        "group_names": group_names,
        "tags": tags,
        "tag_group_masks": masks,
        "base_scores": base_scores,
        "influence": influence,
    }
    _validate_template_dict(template)
    return template


def write_json(json_path: str, template: dict) -> None:
    _validate_template_dict(template)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)


# -------------------------------------------------------------------------
# LOAD / AUTO-GENERATE / CSV-TO-JSON PIPELINE
# -------------------------------------------------------------------------

_MODEL_CACHE: Optional[TagModel] = None
_MODEL_MTIME: Optional[float] = None


def load_tag_model() -> TagModel:
    """
    Behaviour:
      1) Uses the STATIC Python template constants as the default starting values.
      2) Ensures tag_data.json exists (created from the static template if missing).
      3) Ensures tag_data.csv exists (created as an editable template if missing).
      4) If tag_data.csv exists and is newer than tag_data.json, rebuild tag_data.json from CSV.
      5) Load and return tag_data.json (cached by mtime).
    """
    global _MODEL_CACHE, _MODEL_MTIME

    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "tag_data.json")
    csv_path = os.path.join(here, "tag_data.csv")

    static_template = _default_template_dict()

    # Create JSON if missing (from static template)
    if not os.path.exists(json_path):
        write_json(json_path, static_template)

    # Create CSV if missing (from static template)
    if not os.path.exists(csv_path):
        try:
            write_csv_template(csv_path, static_template)
        except Exception as e:
            # Non-fatal; user can still use JSON
            print("[TagModel] Failed to write CSV template:", e)

    # If CSV is newer than JSON, attempt to rebuild JSON from CSV
    try:
        if os.path.exists(csv_path):
            csv_mtime = os.path.getmtime(csv_path)
            json_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else -1
            if csv_mtime > json_mtime:
                tpl = read_template_from_csv(csv_path)
                write_json(json_path, tpl)
    except Exception as e:
        # Non-fatal: keep using the existing JSON if CSV is broken
        print("[TagModel] CSV -> JSON rebuild failed:", e)

    # Cache based on JSON mtime
    mtime = os.path.getmtime(json_path)
    if _MODEL_CACHE is not None and _MODEL_MTIME == mtime:
        return _MODEL_CACHE

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = _dict_to_model(data)
    _MODEL_CACHE = model
    _MODEL_MTIME = mtime
    return model
