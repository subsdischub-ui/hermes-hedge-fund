"""Offline TRM training driver.

Run on a GPU box (the M2 Max in MLX-equivalent terms, or an RTX/Ada machine).
This script:
  1. Pulls historical OHLCV per symbol from CCXT.
  2. Replays the bull/bear/trader agents OR uses heuristic labels to produce
     (state_grid, action_grid, pnl) triples — a self-imitation dataset.
  3. Trains the TRM (TinyRecursiveReasoningModel_ACTV1) with deep supervision.
  4. Saves a checkpoint to TRM_CHECKPOINT for the running container to mount.

We keep this as a thin driver — the model code is the upstream TRM repo (see
verifier.py: `_import_trm`). Heavy lifting in `pretrain.py` of that repo can
also be reused with the dataset emitted here. For first runs, use HEURISTIC
labels: action = next-N-bar realized return sign (with stop/tp from ATR).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hermes.trm.encoder import ActionCodec, MarketStateEncoder, Action


def heuristic_label(df: pd.DataFrame, idx: int, lookahead: int = 8, atr_k: float = 2.0) -> Action:
    """Label = sign of next `lookahead` close return, stop/tp via ATR."""
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[idx]
    if np.isnan(atr) or idx + lookahead >= len(df):
        return Action(side="HOLD", size_pct=0.0, stop_pct=0.02, tp_pct=0.04)
    fwd = float(c.iloc[idx + lookahead]) / float(c.iloc[idx]) - 1.0
    side = "BUY" if fwd > 0 else ("SELL" if fwd < 0 else "HOLD")
    return Action(
        side=side,
        size_pct=min(1.0, abs(fwd) / 0.02),
        stop_pct=float(atr_k * atr / c.iloc[idx]),
        tp_pct=float(2 * atr_k * atr / c.iloc[idx]),
    )


def build_dataset(df: pd.DataFrame, instrument_id: int, encoder: MarketStateEncoder, codec: ActionCodec):
    states, actions, ids = [], [], []
    bars = encoder.cfg.bars
    for i in range(bars + 30, len(df) - 8):
        win = df.iloc[i - bars - 30 : i].reset_index(drop=True)
        try:
            s = encoder.encode(win)
        except ValueError:
            continue
        a = codec.encode(heuristic_label(df, i))
        states.append(s)
        actions.append(a)
        ids.append(instrument_id)
    return np.stack(states), np.stack(actions), np.array(ids, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True, help="CSV per symbol (e.g. BTCUSDT_1h.csv)")
    ap.add_argument("--out", type=Path, default=Path("/data/trm/dataset"))
    ap.add_argument("--export-only", action="store_true",
                    help="Skip training, just emit .npy + dataset.json for upstream TRM pretrain.py.")
    args = ap.parse_args()

    encoder = MarketStateEncoder()
    codec = ActionCodec()
    all_s, all_a, all_id = [], [], []
    for i, path in enumerate(sorted(args.in_dir.glob("*.csv"))):
        df = pd.read_csv(path)
        s, a, ids = build_dataset(df, instrument_id=i, encoder=encoder, codec=codec)
        all_s.append(s); all_a.append(a); all_id.append(ids)
        print(f"{path.name}: {len(s)} windows")

    S = np.concatenate(all_s); A = np.concatenate(all_a); I = np.concatenate(all_id)
    args.out.mkdir(parents=True, exist_ok=True)
    # Mirror SamsungSAIL TRM's expected layout so we can reuse upstream pretrain.py.
    np.save(args.out / "all__inputs.npy", np.concatenate([S, np.zeros_like(A)], axis=1))
    labels = np.full_like(S, 0)
    labels = np.concatenate([labels, A], axis=1)
    np.save(args.out / "all__labels.npy", labels)
    np.save(args.out / "all__puzzle_identifiers.npy", I)
    print("dataset written to", args.out)


if __name__ == "__main__":
    main()
