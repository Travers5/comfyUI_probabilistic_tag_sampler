# 0.07.00
# v0.07.00

from __future__ import annotations

import os
import time
import random
import re
from typing import List, Set, Tuple

from .tag_model import load_tag_model, score_to_probability, combine_scores


# -------------------------------------------------------------------------
# CODE-ONLY DEBUG SETTINGS (edit these directly in the file)
# -------------------------------------------------------------------------

DEBUG_LOG_TO_FILE = True          # Set True to enable log file output.
DEBUG_LOG_APPEND = False          # Append to the same log file (True) or overwrite (False).
DEBUG_LOG_FILENAME = "prob_tag_sampler_debug.log"

# If True, after each chosen tag update, also log updated probabilities (no RNG).
# Warning: can get large quickly.
DEBUG_LOG_INCLUDE_POST_UPDATE_PROBS = False

# If True, also print a short summary to the console (ComfyUI terminal).
PRINT_SUMMARY_TO_CONSOLE = False


# -------------------------------------------------------------------------
# WEIGHT SHAPING + NORMALISATION
# -------------------------------------------------------------------------

# Weight shaping to avoid clustering near the high end
WEIGHT_POS_EXP = 3.0       # higher => more weights closer to 0.50
WEIGHT_FORCED_EXP = 2.0    # shapes forced weights too (below 0.50)

# Optional normalisation of weights to mean 1.0 (post-selection)
NORMALISE_WEIGHTS_DEFAULT = True
NORMALISE_TARGET_MEAN = 1.0


# -------------------------------------------------------------------------
# RNG SHAPE CONTROL
# -------------------------------------------------------------------------

# Default for the UI input
RNG_AVG_COUNT_DEFAULT = 2
# The inclusion RNG uses average of K uniform draws.
# K=1 => uniform.
# K=2 => triangular-ish distribution around 0.5.
# Larger K => stronger concentration around 0.5.


def _safe_group_names() -> List[str]:
    try:
        return list(load_tag_model().group_names)
    except Exception:
        return []


def _ui_key(label: str, effect: str) -> str:
    """
    UI-friendly key: starts with group name.
    """
    return f"{label} ({effect})"


def _bit_for_group(i_1based: int) -> int:
    return 1 << (i_1based - 1)


# --- Strength/weight mapping constants ---
WEIGHT_MIN = 0.35
WEIGHT_THRESHOLD = 0.50
WEIGHT_MAX = 2.00  # as requested

# Influence scaling when strengths are ON:
# factor = weight / 0.50, clamped.
INFLUENCE_SCALE_MIN = 0.10
INFLUENCE_SCALE_MAX = 4.00  # With WEIGHT_MAX=2.0 and threshold=0.5, raw max factor is 4.0.

LOCK_TOL = 1e-12



# -------------------------------------------------------------------------
# INPUT TAG EXTRACTION (chaining / continuation)
# -------------------------------------------------------------------------

# If True, and the input text contains explicit weights like "(tag:1.23)",
# those weights will be used to scale the influence of the pre-selected tags,
# even if output_strengths is Off.
APPLY_INPUT_WEIGHTS_TO_INFLUENCE = True

# If a pre-selected tag is found WITHOUT an explicit weight, we treat its
# influence as "neutral" (factor 1.0), which corresponds to weight 0.50.
PRESELECTED_DEFAULT_WEIGHT = 0.50

def _normalise_weights_mean_constrained(
    weights: List[float],
    *,
    target_mean: float,
    wmin: float,
    wmax: float,
) -> List[float]:
    """
    Scales weights to achieve target_mean while respecting [wmin,wmax].

    If mean too high:
      - keep current maximum fixed first
      - scale others down until mean reached or someone hits wmin
      - if still too high, allow scaling remaining non-wmin weights (including the max)

    If mean too low:
      - keep current minimum fixed first
      - scale others up until mean reached or someone hits wmax
      - if still too low, allow scaling remaining non-wmax weights (including the min)

    Best-effort: if target mean is infeasible given constraints, returns the closest it can.
    """
    if not weights:
        return []

    w = [max(wmin, min(wmax, float(x))) for x in weights]
    n = len(w)

    if n == 1:
        return [max(wmin, min(wmax, target_mean))]

    def mean(vals: List[float]) -> float:
        return sum(vals) / float(len(vals))

    m0 = mean(w)
    if abs(m0 - target_mean) < 1e-9:
        return w

    def scale_once(free: Set[int], fixed: Set[int]) -> None:
        """
        Scale the 'free' set by a factor that would hit the target mean
        assuming 'fixed' stays constant. Caller clamps/moves violators.
        """
        fixed_sum = sum(w[i] for i in fixed)
        free_sum = sum(w[i] for i in free)
        if free_sum <= 1e-12:
            return

        s = (target_mean * n - fixed_sum) / free_sum
        for i in list(free):
            w[i] = w[i] * s

    if m0 > target_mean:
        # Phase 1: keep max fixed, scale others down
        max_idx = max(range(n), key=lambda i: w[i])
        fixed = {max_idx}
        free = set(range(n)) - fixed

        while True:
            scale_once(free, fixed)

            moved = False
            for i in list(free):
                if w[i] < wmin:
                    w[i] = wmin
                    free.remove(i)
                    fixed.add(i)
                    moved = True

            if not moved:
                break

        # Phase 2: if still too high, allow scaling remaining non-wmin weights (including max)
        if mean(w) > target_mean + 1e-9:
            free = {i for i in range(n) if w[i] > wmin + 1e-12}
            fixed = set(range(n)) - free

            while True:
                scale_once(free, fixed)

                moved = False
                for i in list(free):
                    if w[i] < wmin:
                        w[i] = wmin
                        free.remove(i)
                        fixed.add(i)
                        moved = True

                if not moved:
                    break

    else:
        # Phase 1: keep min fixed, scale others up
        min_idx = min(range(n), key=lambda i: w[i])
        fixed = {min_idx}
        free = set(range(n)) - fixed

        while True:
            scale_once(free, fixed)

            moved = False
            for i in list(free):
                if w[i] > wmax:
                    w[i] = wmax
                    free.remove(i)
                    fixed.add(i)
                    moved = True

            if not moved:
                break

        # Phase 2: if still too low, allow scaling remaining non-wmax weights (including min)
        if mean(w) < target_mean - 1e-9:
            free = {i for i in range(n) if w[i] < wmax - 1e-12}
            fixed = set(range(n)) - free

            while True:
                scale_once(free, fixed)

                moved = False
                for i in list(free):
                    if w[i] > wmax:
                        w[i] = wmax
                        free.remove(i)
                        fixed.add(i)
                        moved = True

                if not moved:
                    break

    # Final clamp
    w = [max(wmin, min(wmax, x)) for x in w]
    return w


def _weight_from_margin_positive(margin: float) -> float:
    """
    margin in (0..1]. Map to [0.50..WEIGHT_MAX], but shaped so most values
    stay closer to 0.50 unless margin is very large.
    """
    if margin <= 0.0:
        return WEIGHT_THRESHOLD
    if margin >= 1.0:
        return WEIGHT_MAX

    shaped = margin ** WEIGHT_POS_EXP
    return WEIGHT_THRESHOLD + (WEIGHT_MAX - WEIGHT_THRESHOLD) * shaped


def _weight_from_margin_forced(margin: float) -> float:
    """
    Forced selections: margin <= 0. Map to [0.10..0.49] with shaping.
    """
    if margin <= -1.0:
        return WEIGHT_MIN
    if margin >= 0.0:
        return WEIGHT_THRESHOLD - 0.01

    # closeness: -1 -> 0, 0 -> 1
    closeness = 1.0 + margin
    shaped = closeness ** WEIGHT_FORCED_EXP

    top = WEIGHT_THRESHOLD - 0.01
    return WEIGHT_MIN + (top - WEIGHT_MIN) * shaped


def _weight_for_greedy_min_fill(p: float) -> float:
    """
    Tags added by the post-loop greedy minimum filler were not selected by threshold,
    so keep them < 0.50. Use probability as a mild ordering signal.
    p in [0..1] -> [0.10..0.49]
    """
    if p <= 0.0:
        return WEIGHT_MIN
    if p >= 1.0:
        return WEIGHT_THRESHOLD - 0.01
    return WEIGHT_MIN + (WEIGHT_THRESHOLD - 0.01 - WEIGHT_MIN) * p


def _influence_scale_factor(weight: float) -> float:
    """
    Baseline: weight=0.50 => factor=1.0
    Larger weights => stronger influence; smaller => weaker.
    """
    if WEIGHT_THRESHOLD <= 0:
        return 1.0
    raw = float(weight) / WEIGHT_THRESHOLD
    if raw < INFLUENCE_SCALE_MIN:
        return INFLUENCE_SCALE_MIN
    if raw > INFLUENCE_SCALE_MAX:
        return INFLUENCE_SCALE_MAX
    return raw


def _scale_delta(delta: float, factor: float) -> float:
    """
    Do not scale exact lock influences (+1/-1).
    """
    if abs(delta - 1.0) <= LOCK_TOL or abs(delta + 1.0) <= LOCK_TOL:
        return delta
    return delta * factor


def _sample_r_avg(rng: random.Random, k: int) -> Tuple[float, List[float]]:
    """
    Returns (avg, samples).
    """
    if k <= 1:
        r = rng.random()
        return r, [r]
    samples = [rng.random() for _ in range(k)]
    return sum(samples) / float(k), samples


def _open_debug_log() -> Tuple[bool, str, object]:
    """
    Opens the debug log file if enabled.
    Returns (enabled, path, file_handle).
    """
    if not DEBUG_LOG_TO_FILE:
        return False, "", None

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, DEBUG_LOG_FILENAME)
    mode = "a" if DEBUG_LOG_APPEND else "w"
    f = open(path, mode, encoding="utf-8", newline="\n")

    # Session header
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write(f"Session start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")
    return True, path, f


def _normalise_tag_key(s: str) -> str:
    # Lowercase + collapse whitespace so "long   hair" matches "long hair"
    return " ".join(s.strip().lower().split())


def _try_parse_tag_and_weight(token: str) -> Tuple[str, float | None]:
    """
    Parses a single prompt token into (tag, weight).

    Accepts:
      - "tag"
      - "(tag)"
      - "(tag:1.23)"
      - "tag:1.23"

    If the trailing portion after the final ':' is not a float, it is treated as part of the tag.
    """
    t = token.strip()
    if not t:
        return "", None

    # Remove a single outer pair of parentheses (common Comfy prompt style)
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    # Strip a single pair of matching quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("\'") and t.endswith("\'")):
        t = t[1:-1].strip()
    if ":" in t:
        left, right = t.rsplit(":", 1)
        try:
            w = float(right.strip())
            tag = left.strip()
            if tag:
                return tag, w
        except ValueError:
            pass

    return t, None


def _find_preselected_tags(
    input_text: str,
    model,
    excluded: Set[int],
    *,
    log_f=None,
) -> List[Tuple[int, float | None]]:
    """
    Finds any known tags inside input_text and returns them in order of appearance.

    Strategy:
      1) Explicit weighted occurrences like "(tag:1.23)" anywhere in the text
      2) Comma/newline/; /| separated tokens (with optional ":weight")
      3) Fallback boundary scan across the full text for any remaining tags

    Excluded tags (score <= -1 after group filtering) are ignored.
    Duplicate tags are returned only once (first occurrence wins).
    """
    if not input_text or not input_text.strip():
        if log_f is not None:
            log_f.write("PRESELECTED FROM INPUT_TEXT: none (empty input)\n\n")
        return []

    norm_to_idx = {_normalise_tag_key(name): i for i, name in enumerate(model.tag_names)}

    hits: List[Tuple[int, int, float | None]] = []

    # 1) Find explicit "(...:...)" patterns anywhere
    for m in re.finditer(r"\(\s*([^\(\)]*?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\)", input_text):
        tag_raw = m.group(1)
        try:
            w = float(m.group(2))
        except ValueError:
            continue

        key = _normalise_tag_key(tag_raw)
        idx = norm_to_idx.get(key)
        if idx is not None:
            hits.append((m.start(), idx, w))

    # 2) Parse comma/newline/; /| separated tokens (preserve positions)
    for m in re.finditer(r"[^,\n\r;|]+", input_text):
        token = m.group(0).strip()
        if not token:
            continue

        tag_raw, w = _try_parse_tag_and_weight(token)
        if not tag_raw:
            continue

        key = _normalise_tag_key(tag_raw)
        idx = norm_to_idx.get(key)
        if idx is not None:
            hits.append((m.start(), idx, w))

    # 3) Fallback: boundary scan for any remaining tags inside free-form text
    found_idxs = {idx for _, idx, _ in hits}

    # Prefer longer tags first to reduce partial/substring matches
    tag_order = sorted(range(len(model.tag_names)), key=lambda i: len(model.tag_names[i]), reverse=True)

    for idx in tag_order:
        if idx in found_idxs:
            continue

        tag = model.tag_names[idx]
        if not tag:
            continue

        esc = re.escape(tag)
        # "word-ish" boundary: avoid matching inside other words like "cat" in "catch"
        pat = rf"(?i)(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])"
        mm = re.search(pat, input_text)
        if mm:
            hits.append((mm.start(), idx, None))

    # Sort by position and dedupe by tag index
    hits.sort(key=lambda x: x[0])

    out: List[Tuple[int, float | None]] = []
    seen: Set[int] = set()
    for _, idx, w in hits:
        if idx in excluded:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, w))

    if log_f is not None:
        if out:
            log_f.write("PRESELECTED FROM INPUT_TEXT:\n")
            for idx, w in out:
                if w is None:
                    log_f.write(f"  {model.tag_names[idx]}\n")
                else:
                    log_f.write(f"  {model.tag_names[idx]} (w={w:.3f})\n")
            log_f.write("\n")
        else:
            log_f.write("PRESELECTED FROM INPUT_TEXT: none\n\n")

    return out


def _apply_tag_influence_update(
    scores: List[float],
    chosen_idx: int,
    model,
    *,
    factor: float,
    excluded: Set[int],
    selected_set: Set[int],
) -> None:
    """
    Applies the chosen tag's influence row onto all remaining (non-excluded, non-selected) tags.
    """
    row = model.influence[chosen_idx]
    n = len(scores)

    for j in range(n):
        if j in excluded or j in selected_set:
            continue

        delta = row[j]
        if factor != 1.0:
            delta = _scale_delta(delta, factor)

        # Clamp to [-1, 1]
        if delta > 1.0:
            delta = 1.0
        elif delta < -1.0:
            delta = -1.0

        scores[j] = combine_scores(scores[j], delta)


class ProbabilisticTagSampler:
    CATEGORY = "text/tags"

    @classmethod
    def INPUT_TYPES(cls):
        group_names = _safe_group_names()

        required = {
            # NOTE:
            # ComfyUI's UI treats an input named 'seed' specially and often renders its own
            # "control after generate" option automatically. Therefore we do NOT define a
            # separate 'control_after_generate' input here (avoids duplicate controls).
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "input_text": ("STRING", {"default": "", "multiline": True}),

            "min_tags": ("INT", {"default": 0, "min": 0, "max": 9999}),
            "max_tags": ("INT", {"default": -1, "min": -1, "max": 9999}),

            "separator": (["comma_space", "comma", "space", "newline"], {"default": "comma_space"}),

            "boost_amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "suppress_amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),

            "output_strengths": ("BOOLEAN", {"default": False, "label_on": "Strengths On", "label_off": "Strengths Off"}),

            "rng_avg_count": ("INT", {"default": RNG_AVG_COUNT_DEFAULT, "min": 1, "max": 32}),
            "normalise_strengths": ("BOOLEAN", {"default": NORMALISE_WEIGHTS_DEFAULT,
                                                "label_on": "Normalise strengths",
                                                "label_off": "Raw strengths"}),
        }

        # Per-group: Exclude / Boost / Suppress
        for label in group_names:
            required[_ui_key(label, "Exclude")] = ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"})
            required[_ui_key(label, "Boost")] = ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"})
            required[_ui_key(label, "Suppress")] = ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"})

        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "generate"

    def generate(
        self,
        seed: int,
        input_text: str,
        min_tags: int,
        max_tags: int,
        separator: str,
        boost_amount: float,
        suppress_amount: float,
        output_strengths: bool,
        rng_avg_count: int,
        normalise_strengths: bool,
        **kwargs,
    ):
        model = load_tag_model()
        n = model.size()
        group_names = list(model.group_names)
        m = len(group_names)

        # Open debug log if enabled
        log_enabled, log_path, log_f = _open_debug_log()
        try:
            if log_enabled:
                log_f.write(f"seed={seed}\n")
                log_f.write(f"rng_avg_count={int(rng_avg_count)}\n")
                log_f.write(f"min_tags={min_tags}, max_tags={max_tags}\n")
                log_f.write(f"output_strengths={output_strengths}\n")
                log_f.write(f"normalise_strengths={normalise_strengths}\n")
                log_f.write(f"WEIGHT_MAX={WEIGHT_MAX}\n")
                log_f.write(f"WEIGHT_POS_EXP={WEIGHT_POS_EXP}, WEIGHT_FORCED_EXP={WEIGHT_FORCED_EXP}\n")
                log_f.write("\n")

            # Determine group effect bitmasks
            exclude_bits = 0
            boost_bits = 0
            suppress_bits = 0

            for i in range(1, m + 1):
                label = group_names[i - 1]
                bit = _bit_for_group(i)

                if bool(kwargs.get(_ui_key(label, "Exclude"), False)):
                    exclude_bits |= bit
                if bool(kwargs.get(_ui_key(label, "Boost"), False)):
                    boost_bits |= bit
                if bool(kwargs.get(_ui_key(label, "Suppress"), False)):
                    suppress_bits |= bit

            # Working scores
            scores = list(model.base_scores)

            # Apply group effects (exclude hard lock; boost/suppress as deltas)
            for idx in range(n):
                tag_mask = model.tag_group_masks[idx]

                if (tag_mask & exclude_bits) != 0:
                    scores[idx] = -1.0
                    continue

                if (tag_mask & boost_bits) != 0 and boost_amount > 0.0:
                    scores[idx] = combine_scores(scores[idx], float(boost_amount))
                if (tag_mask & suppress_bits) != 0 and suppress_amount > 0.0:
                    scores[idx] = combine_scores(scores[idx], -float(suppress_amount))

            excluded: Set[int] = {i for i, s in enumerate(scores) if s <= -1.0}

            rng = random.Random(int(seed))

            selected: List[int] = []
            selected_weights: List[float] = []

            remaining_cap = max(0, n - len(excluded))
            if max_tags is None or max_tags < 0:
                hard_max = remaining_cap
            else:
                hard_max = min(int(max_tags), remaining_cap)

            effective_min = min(int(min_tags), hard_max)

            # -----------------------------------------------------------------
            # Pre-select tags already present in the provided input_text.
            # This enables chaining: you can feed this node's output into another
            # instance, or pass in a larger text blob that already contains tags.
            # -----------------------------------------------------------------

            preselected = _find_preselected_tags(
                input_text=input_text,
                model=model,
                excluded=excluded,
                log_f=log_f if log_enabled else None,
            )

            selected_set: Set[int] = set()

            # Add preselected tags up to hard_max
            if preselected:
                for idx, w_in in preselected:
                    if len(selected) >= hard_max:
                        break

                    selected.append(idx)
                    selected_set.add(idx)

                    # Store a weight for output (if strengths are on) so the preselected
                    # tags remain visible and chainable.
                    if output_strengths:
                        w_store = float(w_in) if w_in is not None else float(PRESELECTED_DEFAULT_WEIGHT)
                    else:
                        w_store = 1.0
                    selected_weights.append(w_store)

                # Apply influence updates for each preselected tag in order.
                # If a preselected tag came with an explicit weight, optionally use
                # it to scale influence (so chaining preserves behaviour).
                for idx, w_in in preselected:
                    if idx not in selected_set:
                        continue

                    factor = 1.0
                    if APPLY_INPUT_WEIGHTS_TO_INFLUENCE and (w_in is not None):
                        factor = _influence_scale_factor(float(w_in))

                    _apply_tag_influence_update(
                        scores,
                        idx,
                        model,
                        factor=factor,
                        excluded=excluded,
                        selected_set=selected_set,
                    )

            # If the input already has >= hard_max tags, we cannot add any more.
            if len(selected) >= hard_max:
                # Skip the main loop entirely; normalisation/output still apply.
                pass


            if log_enabled:
                log_f.write(f"exclude_bits={exclude_bits}, boost_bits={boost_bits}, suppress_bits={suppress_bits}\n")
                log_f.write(f"excluded_count={len(excluded)} hard_max={hard_max} effective_min={effective_min}\n\n")

            iteration = 0

            # Main selection loop
            while len(selected) < hard_max:
                iteration += 1

                best_pos_idx = None
                best_pos_margin = -1e18
                best_pos_p = 0.0
                best_pos_ravg = 0.0
                best_pos_samples: List[float] = []

                best_neg_idx = None
                best_neg_margin = -1e18
                best_neg_p = 0.0
                best_neg_ravg = 0.0
                best_neg_samples: List[float] = []

                any_candidate = False

                if log_enabled:
                    log_f.write("-" * 80 + "\n")
                    log_f.write(f"ITERATION {iteration} | selected={len(selected)}\n")
                    log_f.write("tag\tscore\tprob\tr_avg\tr_samples\tmargin\n")

                # Evaluate all candidates
                for i in range(n):
                    if i in excluded or i in selected_set:
                        continue
                    any_candidate = True

                    p = score_to_probability(scores[i])
                    r_avg, r_samples = _sample_r_avg(rng, int(rng_avg_count))
                    margin = p - r_avg

                    if log_enabled:
                        log_f.write(
                            f"{model.tag_names[i]}\t"
                            f"{scores[i]: .6f}\t"
                            f"{p: .6f}\t"
                            f"{r_avg: .6f}\t"
                            f"{','.join(f'{x:.6f}' for x in r_samples)}\t"
                            f"{margin: .6f}\n"
                        )

                    if margin > 0:
                        if margin > best_pos_margin:
                            best_pos_margin = margin
                            best_pos_idx = i
                            best_pos_p = p
                            best_pos_ravg = r_avg
                            best_pos_samples = r_samples
                    else:
                        if margin > best_neg_margin:
                            best_neg_margin = margin
                            best_neg_idx = i
                            best_neg_p = p
                            best_neg_ravg = r_avg
                            best_neg_samples = r_samples

                if not any_candidate:
                    if log_enabled:
                        log_f.write("No candidates remain. Stopping.\n")
                    break

                forced = False
                if best_pos_idx is not None:
                    chosen = best_pos_idx
                    chosen_margin = best_pos_margin
                    chosen_p = best_pos_p
                    chosen_ravg = best_pos_ravg
                    chosen_samples = best_pos_samples
                else:
                    if len(selected) >= effective_min:
                        if log_enabled:
                            log_f.write("No margins > 0 and min_tags already satisfied. Stopping.\n")
                        break
                    if best_neg_idx is None:
                        if log_enabled:
                            log_f.write("No negative candidate available (unexpected). Stopping.\n")
                        break
                    chosen = best_neg_idx
                    chosen_margin = best_neg_margin
                    chosen_p = best_neg_p
                    chosen_ravg = best_neg_ravg
                    chosen_samples = best_neg_samples
                    forced = True

                # Compute weight (only meaningful when strengths are ON)
                if output_strengths:
                    if (not forced) and chosen_margin > 0:
                        w = _weight_from_margin_positive(chosen_margin)
                    else:
                        w = _weight_from_margin_forced(chosen_margin)
                else:
                    w = 1.0

                selected.append(chosen)
                selected_set.add(chosen)
                selected_weights.append(w)

                # Influence scaling factor
                if output_strengths:
                    factor = _influence_scale_factor(w)
                else:
                    factor = 1.0

                if log_enabled:
                    log_f.write("\nCHOSEN:\n")
                    log_f.write(f"  tag={model.tag_names[chosen]}\n")
                    log_f.write(f"  forced={forced}\n")
                    log_f.write(f"  score={scores[chosen]:.6f}\n")
                    log_f.write(f"  prob={chosen_p:.6f}\n")
                    log_f.write(f"  r_avg={chosen_ravg:.6f}\n")
                    log_f.write(f"  r_samples={','.join(f'{x:.6f}' for x in chosen_samples)}\n")
                    log_f.write(f"  margin={chosen_margin:.6f}\n")
                    log_f.write(f"  weight={w:.6f}\n")
                    log_f.write(f"  influence_scale_factor={factor:.6f}\n\n")

                # Apply influence updates (scaled if strengths are ON)
                row = model.influence[chosen]
                for j in range(n):
                    if j in excluded or j in selected_set:
                        continue

                    delta = row[j]
                    if output_strengths and factor != 1.0:
                        delta = _scale_delta(delta, factor)

                    # Clamp scaled delta to [-1, 1]
                    if delta > 1.0:
                        delta = 1.0
                    elif delta < -1.0:
                        delta = -1.0

                    scores[j] = combine_scores(scores[j], delta)

                if log_enabled and DEBUG_LOG_INCLUDE_POST_UPDATE_PROBS:
                    log_f.write("POST-UPDATE PROBABILITIES:\n")
                    log_f.write("tag\tscore\tprob\n")
                    for i in range(n):
                        if i in excluded or i in selected_set:
                            continue
                        p = score_to_probability(scores[i])
                        log_f.write(f"{model.tag_names[i]}\t{scores[i]: .6f}\t{p: .6f}\n")
                    log_f.write("\n")

            # Enforce minimum count if short: greedily add highest probability remaining tags
            if len(selected) < effective_min:
                candidates: List[Tuple[float, int]] = []
                for i in range(n):
                    if i in excluded or i in selected_set:
                        continue
                    p = score_to_probability(scores[i])
                    candidates.append((p, i))
                candidates.sort(reverse=True)

                if log_enabled:
                    log_f.write("-" * 80 + "\n")
                    log_f.write(f"MIN-FILL (needed {effective_min - len(selected)} more tags)\n")

                for p, i in candidates:
                    if len(selected) >= effective_min:
                        break
                    selected.append(i)
                    selected_set.add(i)
                    if output_strengths:
                        w = _weight_for_greedy_min_fill(p)
                    else:
                        w = 1.0
                    selected_weights.append(w)

                    if log_enabled:
                        log_f.write(f"  added={model.tag_names[i]} prob={p:.6f} weight={w:.6f}\n")

            # -----------------------------------------------------------------
            # Strength normalisation (mean -> 1.0), after selection but before output
            # -----------------------------------------------------------------
            if output_strengths and normalise_strengths and selected_weights:
                before_mean = sum(selected_weights) / float(len(selected_weights))

                selected_weights = _normalise_weights_mean_constrained(
                    selected_weights,
                    target_mean=NORMALISE_TARGET_MEAN,
                    wmin=WEIGHT_MIN,
                    wmax=WEIGHT_MAX,
                )

                after_mean = sum(selected_weights) / float(len(selected_weights))

                if log_enabled:
                    log_f.write("\nWEIGHT NORMALISATION:\n")
                    log_f.write(f"  before_mean={before_mean:.6f}\n")
                    log_f.write(f"  after_mean ={after_mean:.6f}\n")
                    log_f.write("  tag_weights=" + ",".join(
                        f"{model.tag_names[i]}:{w:.3f}" for i, w in zip(selected, selected_weights)
                    ) + "\n")

            # Output formatting
            if separator == "comma_space":
                sep = ", "
            elif separator == "comma":
                sep = ","
            elif separator == "space":
                sep = " "
            else:
                sep = "\n"

            if not output_strengths:
                out = sep.join(model.tag_names[i] for i in selected)
            else:
                parts = []
                for idx, w in zip(selected, selected_weights):
                    # final clamp for safety
                    if w < WEIGHT_MIN:
                        w = WEIGHT_MIN
                    if w > WEIGHT_MAX:
                        w = WEIGHT_MAX
                    parts.append(f"({model.tag_names[idx]}:{w:.2f})")
                out = sep.join(parts)

            if log_enabled:
                log_f.write("\nFINAL OUTPUT:\n")
                log_f.write(out + "\n")

            if PRINT_SUMMARY_TO_CONSOLE:
                msg = f"[ProbabilisticTagSampler] selected={len(selected)}"
                if log_enabled:
                    msg += f" log={log_path}"
                print(msg)

            return (out,)

        finally:
            if log_enabled and log_f is not None:
                log_f.flush()
                log_f.close()
