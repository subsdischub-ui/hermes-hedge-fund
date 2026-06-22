"""Encode market state -> fixed-size discrete token grid for TRM.

Layout (default cfg):
    state_len   = 8 channels x 32 bars = 256 tokens
    action_len  = 4 cells: [side, size_bkt, stop_bkt, tp_bkt]
    seq_len     = state_len + action_len = 260

Vocab map (single shared vocab so the model head never needs resizing):
       0          PAD / IGNORE_LABEL_ID
       1-12       SIDE        (HOLD, BUY, SELL + reserved)
      13-28       SIZE bucket (16 size levels)
      29-60       STOP bucket (32 stop distances)
      61-92       TP bucket   (32 tp distances)
      93-95       reserved
      96-159      STATE bins  (64 quantile bins for state cells)
      vocab_size = 160

The state region (96-159) is REQUIRED to be disjoint from the action region
(0-95). Earlier versions reused low ids for state, which caused state bin 0
to collide with PAD_ID and state bin 1 with SIDE_HOLD — both broke the
TRM forward pass semantically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- vocab regions ---------------------------------------------------------
PAD_ID = 0
SIDE_BASE = 1
SIDE_HOLD, SIDE_BUY, SIDE_SELL = SIDE_BASE, SIDE_BASE + 1, SIDE_BASE + 2
SIZE_BASE = 13
N_SIZE_BUCKETS = 16
STOP_BASE = 29
N_STOP_BUCKETS = 32
TP_BASE = 61
N_TP_BUCKETS = 32

STATE_BASE = 96
N_STATE_BINS = 64
VOCAB_SIZE = STATE_BASE + N_STATE_BINS   # 160


# --- state encoding --------------------------------------------------------
DEFAULT_CHANNELS: tuple[str, ...] = (
    "ret_1",       # log return 1 bar
    "ret_log",     # cumulative log return (mean-centered)
    "vol_z",       # ATR / price, z-scored
    "rsi_14",
    "ema_fast_diff",  # (close - ema8) / atr
    "ema_slow_diff",  # (close - ema21) / atr
    "vol_ratio",      # volume / 20-bar ma
    "regime_bit",     # 1 if trending, else 0 (HHV/LLV breakout flag)
)


@dataclass(frozen=True)
class EncoderConfig:
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    bars: int = 32
    state_bins: int = N_STATE_BINS

    @property
    def state_len(self) -> int:
        return len(self.channels) * self.bars


class MarketStateEncoder:
    """Quantize an OHLCV DataFrame into a fixed-size int32 token grid.

    Quantization uses a wider rolling window than the encoded slice itself
    (`bars + lookback`) to keep bin boundaries stable across consecutive ticks.
    """

    LOOKBACK_FOR_QUANTILES = 200

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute channel features from an OHLCV frame. Returns the last `bars + lookback` rows."""
        min_required = self.cfg.bars + 30
        if len(df) < min_required:
            raise ValueError(f"need at least {min_required} bars, got {len(df)}")
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        h, l = df["high"].astype(float), df["low"].astype(float)

        tr = pd.concat(
            [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        ema_fast = c.ewm(span=8, adjust=False).mean()
        ema_slow = c.ewm(span=21, adjust=False).mean()

        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = (up / down.replace(0, np.nan)).fillna(0)
        rsi = 100 - (100 / (1 + rs))

        vol_z_num = (atr / c) - (atr / c).rolling(50, min_periods=10).mean()
        vol_z_den = (atr / c).rolling(50, min_periods=10).std().replace(0, np.nan)
        vol_z = vol_z_num / vol_z_den

        feat = pd.DataFrame(index=df.index)
        feat["ret_1"] = np.log(c / c.shift(1))
        feat["ret_log"] = feat["ret_1"].rolling(20).sum()
        feat["vol_z"] = vol_z
        feat["rsi_14"] = rsi
        feat["ema_fast_diff"] = (c - ema_fast) / atr.replace(0, np.nan)
        feat["ema_slow_diff"] = (c - ema_slow) / atr.replace(0, np.nan)
        feat["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
        feat["regime_bit"] = (c > c.rolling(50).max().shift(1)).astype(float)
        return feat.tail(self.cfg.bars + self.LOOKBACK_FOR_QUANTILES)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (state_len,) int32 array of token IDs in [STATE_BASE, VOCAB_SIZE)."""
        f = self.features(df).fillna(0.0)
        out = np.zeros((self.cfg.bars, len(self.cfg.channels)), dtype=np.int32)
        for j, name in enumerate(self.cfg.channels):
            col = f[name].to_numpy()
            bins = _quantize(col, self.cfg.state_bins)
            # Slice down to the last `bars` rows (we encoded extra context for
            # stable bin boundaries).
            out[:, j] = STATE_BASE + bins[-self.cfg.bars :]
        return out.reshape(-1).astype(np.int32)


def _quantize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-rank → integer bin in [0, n_bins-1]."""
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy()
    bins = np.clip((ranks * n_bins).astype(np.int32), 0, n_bins - 1)
    return bins


# --- action codec ----------------------------------------------------------
@dataclass
class Action:
    side: str            # "BUY" | "SELL" | "HOLD"
    size_pct: float      # 0..1 of stage cap
    stop_pct: float      # stop distance as % of entry
    tp_pct: float        # take-profit distance as % of entry


class ActionCodec:
    """Encode/decode Action <-> 4 token cells."""

    SIDE_TO_ID = {"HOLD": SIDE_HOLD, "BUY": SIDE_BUY, "SELL": SIDE_SELL}
    ID_TO_SIDE = {v: k for k, v in SIDE_TO_ID.items()}

    # Log-spaced grids so small stop/TP values are NOT silently rounded up
    # to several times their true magnitude.
    STOP_GRID = np.geomspace(0.0005, 0.10, N_STOP_BUCKETS)   # 0.05% .. 10%
    TP_GRID = np.geomspace(0.001, 0.20, N_TP_BUCKETS)        # 0.1% .. 20%

    def encode(self, a: Action) -> np.ndarray:
        side_id = self.SIDE_TO_ID.get(a.side.upper(), SIDE_HOLD)
        size_b = int(np.clip(round(a.size_pct * (N_SIZE_BUCKETS - 1)), 0, N_SIZE_BUCKETS - 1))
        stop_b = int(np.argmin(np.abs(np.log(self.STOP_GRID) - np.log(max(a.stop_pct, 1e-6)))))
        tp_b = int(np.argmin(np.abs(np.log(self.TP_GRID) - np.log(max(a.tp_pct, 1e-6)))))
        return np.array(
            [side_id, SIZE_BASE + size_b, STOP_BASE + stop_b, TP_BASE + tp_b], dtype=np.int32
        )

    def decode(self, tokens: np.ndarray) -> Action:
        side_id, size_t, stop_t, tp_t = int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])
        side = self.ID_TO_SIDE.get(side_id, "HOLD")
        size_b = max(0, min(N_SIZE_BUCKETS - 1, size_t - SIZE_BASE))
        stop_b = max(0, min(N_STOP_BUCKETS - 1, stop_t - STOP_BASE))
        tp_b = max(0, min(N_TP_BUCKETS - 1, tp_t - TP_BASE))
        return Action(
            side=side,
            size_pct=size_b / (N_SIZE_BUCKETS - 1),
            stop_pct=float(self.STOP_GRID[stop_b]),
            tp_pct=float(self.TP_GRID[tp_b]),
        )
