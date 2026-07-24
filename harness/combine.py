"""Combine multiple cross-sectional signals into one composite and attribute each
signal's value via leave-one-out incremental IC.

A signal that is predictive but *redundant* with others adds little once they are
present -- exactly what standalone IC hides and incremental IC reveals. This is
the tool for deciding whether the 10-K text signal or a Kronos forecast actually
adds anything on top of momentum, rather than just correlating with returns.

Combination: z-score each signal cross-sectionally (per day), orient it by the
sign of its IC, weight, and sum. Missing values impute to a neutral 0 so signals
with different coverage still blend.

Orientation uses full-sample IC, a mild in-sample choice; in production, fix each
signal's sign on a training window before combining out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import rank_ic


def cross_sectional_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (day) to mean 0, std 1 across stocks."""
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0.0, np.nan)
    return signal.sub(mu, axis=0).div(sd, axis=0)


def _common_grid(frames: list[pd.DataFrame]):
    idx = cols = None
    for f in frames:
        idx = f.index if idx is None else idx.intersection(f.index)
        cols = f.columns if cols is None else cols.intersection(f.columns)
    return idx.sort_values(), cols.sort_values()


def combine_signals(
    signals: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Blend `signals` into one composite frame.

    Each signal is z-scored, multiplied by its sign (from `signs`, else the sign
    of its IC vs `fwd_ret`, else +1) and its weight (from `weights`, else equal),
    then summed. Cells where every signal is missing stay NaN.
    """
    idx, cols = _common_grid(list(signals.values()))
    aligned = {n: s.reindex(index=idx, columns=cols) for n, s in signals.items()}
    names = list(aligned)

    if signs is None:
        if fwd_ret is not None:
            signs = {n: (1.0 if rank_ic(aligned[n], fwd_ret).mean() >= 0 else -1.0) for n in names}
        else:
            signs = {n: 1.0 for n in names}
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    combo = sum(
        (cross_sectional_zscore(aligned[n]) * signs[n] * weights[n]).fillna(0.0)
        for n in names
    )
    present = None
    for n in names:
        p = aligned[n].notna()
        present = p if present is None else (present | p)
    return combo.where(present)


def incremental_ic(
    signals: dict[str, pd.DataFrame], fwd_ret: pd.DataFrame
) -> dict:
    """Attribute each signal's contribution to the equal-weight composite.

    Returns {'combined_ic': float, 'attribution': DataFrame[standalone_ic,
    incremental_ic]}, where incremental_ic[n] = combined IC minus the IC of the
    composite built without signal n (its leave-one-out marginal value). A high
    standalone but low incremental IC means the signal is redundant with others.
    """
    names = list(signals)
    combined_ic = rank_ic(combine_signals(signals, fwd_ret), fwd_ret).mean()

    rows = {}
    for n in names:
        standalone = rank_ic(signals[n], fwd_ret).mean()
        if len(names) > 1:
            rest = {k: v for k, v in signals.items() if k != n}
            ic_without = rank_ic(combine_signals(rest, fwd_ret), fwd_ret).mean()
            incremental = combined_ic - ic_without
        else:
            incremental = combined_ic
        rows[n] = {"standalone_ic": standalone, "incremental_ic": incremental}

    attribution = pd.DataFrame(rows).T[["standalone_ic", "incremental_ic"]].astype(float)
    return {"combined_ic": float(combined_ic), "attribution": attribution}
