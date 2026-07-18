"""Shadow measurement of the maker fill-rate on OUR OWN live entries (P1).

WHY THIS EXISTS — the frozen pre-registration: ``research_maker/PREREG_SHADOW_FILL.md``
(read it before touching the semantics below; nothing here is a judgment call, it is a
mechanical transcription of that document). In one sentence: every time the paper engine
opens a LIVE entry (taker, today), we ALSO log a hypothetical maker limit order at the
same decision moment and check, on the next 4h bar, whether it would have filled — so we
can measure, never assume, whether flipping ``maker_orders: true`` would pay for itself.

This module does NOT execute anything, place any real/paper order, or touch
``maker_orders``. It is pure shadow accounting: read-modify-write one JSON-lines file.

Persistence model
------------------
No in-memory state survives between calls — every public function reads the current
file, mutates, and writes it back (atomic tmp + os.replace, mirroring
``hermes.execution.paper.PaperExchange``). This makes "restart" trivially safe: there is
nothing to restart, the file IS the state. Each line is the CURRENT full record for one
shadow order (not a raw append-only event log) so ``resolve_pending`` can update a
pending record's resolution fields in place.

Fill semantics (gelées, prereg) — for a BUY, limit = decision_close * (1 - offset):
  * STRICT (honest lower bound): next bar's ``low`` must go STRICTLY BELOW the limit —
    an exact touch does NOT count. Symmetric for a SELL (``high`` strictly above).
  * AU-TOUCHER (upper bound): ``low <= limit`` / ``high >= limit``.
Both offsets (0 and 5 bps) are always computed; both fill flavours are always computed.

Chase cost — only defined for a STRICT(offset 0) non-fill: the adverse price move
between the decision close and the next bar's ``open`` (what re-entering taker, late,
would have cost). It is the RAW empirical value (can be negative i.e. favourable), never
modeled or clipped — the prereg's "moyenne EMPIRIQUE" language is explicit about this.

Verdict — evaluated once 40 signals have RESOLVED, using exactly the first 40 (by
entry order, which is preserved since this module only ever appends new records and
updates existing ones in place — never reorders or deletes). Because that slice never
changes identity as more signals accrue, ``verdict_at_40`` is naturally frozen forever
once first computed: no caching flag is needed, and no code path re-reads a wider or
shifted window (prereg: "Pas de 2e fenêtre, pas de seuil révisé après lecture").
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)

OFFSETS_BPS: tuple[float, ...] = (0.0, 5.0)
GAIN_PAR_FILL_BPS = 3.0  # perp_taker - perp_maker, entry leg (frozen, prereg)
VERDICT_N = 40
VERDICT_FILL_STRICT_MIN = 0.70
VERDICT_GAIN_NET_MIN_BPS = 2.0

PREREG_PATH = "research_maker/PREREG_SHADOW_FILL.md"

_DIRNAME = "maker_shadow"
_FILENAME = "log.jsonl"


# ---- path / io helpers -----------------------------------------------------
def _base_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return Path(get_settings().data_dir)


def _log_path(data_dir: Path | str | None = None) -> Path:
    return _base_dir(data_dir) / _DIRNAME / _FILENAME


def _load(data_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Read the current shadow log — empty list if never written or unreadable.
    A corrupt individual line is skipped (never fabricated, never fatal)."""
    path = _log_path(data_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("shadow_maker_load_failed", path=str(path), error=str(e))
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        with contextlib.suppress(Exception):
            out.append(json.loads(line))
    return out


def _save(entries: list[dict[str, Any]], data_dir: Path | str | None = None) -> None:
    """Atomically rewrite the full log (tmp + os.replace), one JSON object per line."""
    path = _log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=_FILENAME + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, default=str))
                f.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(tmp)
        raise


# ---- pure fill-price math ---------------------------------------------------
def _limits_for(side: str, decision_close: float) -> dict[float, float]:
    """Maker limit price per offset (bps) — below close for a buy, above for a sell."""
    side_u = (side or "").upper()
    out: dict[float, float] = {}
    for off in OFFSETS_BPS:
        if side_u == "BUY":
            out[off] = decision_close * (1 - off / 10_000.0)
        else:
            out[off] = decision_close * (1 + off / 10_000.0)
    return out


# ---- public API ---------------------------------------------------------
BAR_INTERVAL_MS = 4 * 60 * 60 * 1000  # the slate decides on 4h bars (config decision_timeframe)
# Tolerance on "is this THE next bar": venue timestamps land exactly on the 4h grid,
# but allow drift well under a bar so a slightly-late fetch never voids a real next bar.
_NEXT_BAR_TOLERANCE_MS = 10 * 60 * 1000


def record_entry(
    ts: Any,
    symbol: str,
    side: str,
    skill_id: str,
    decision_close: float,
    *,
    decision_bar_ts: int | None = None,
    data_dir: Path | str | None = None,
) -> None:
    """Log a hypothetical maker order for a just-executed LIVE entry.

    ``decision_bar_ts`` is the epoch-ms OPEN timestamp of the (closed) decision bar —
    it anchors the anti-gap guard in ``resolve_pending`` (only THE immediately-next
    bar may resolve this entry). ``None`` is tolerated fail-soft (the entry then
    resolves without the guard) but the live hooks always pass it.

    Fail-soft by construction: never raises. A disk/permission problem degrades to a
    logged warning — this module must never break the market tick it observes.
    """
    try:
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        decision_close = float(decision_close)
        side_u = (side or "").upper()
        limits = _limits_for(side_u, decision_close)
        entry: dict[str, Any] = {
            "id": f"{symbol}|{skill_id}|{ts_iso}|{uuid4().hex[:8]}",
            "ts": ts_iso,
            "symbol": symbol,
            "side": side_u,
            "skill_id": skill_id or "",
            "decision_close": decision_close,
            "decision_bar_ts": int(decision_bar_ts) if decision_bar_ts is not None else None,
            "limit_0": limits[0.0],
            "limit_5": limits[5.0],
            "resolved": False,
            "void": False,
            "fill_strict_0": None,
            "fill_touch_0": None,
            "fill_strict_5": None,
            "fill_touch_5": None,
            "chase_cost_bps": None,
        }
        entries = _load(data_dir)
        entries.append(entry)
        _save(entries, data_dir)
    except Exception as e:  # pragma: no cover - defensive, exercised via readonly-dir test
        log.warning("shadow_maker_record_entry_failed", symbol=symbol, error=str(e))


def resolve_pending(
    symbol: str,
    next_bar_ohlc: dict[str, float],
    *,
    bar_ts: int | None = None,
    data_dir: Path | str | None = None,
) -> None:
    """Resolve every PENDING shadow entry for ``symbol`` against the next 4h bar.

    ``next_bar_ohlc`` is the LATEST CLOSED bar as of this tick ; ``bar_ts`` is its
    epoch-ms OPEN timestamp. GARDE ANTI-ÉCART (audit 2026-07-18) : la règle gelée du
    prereg définit le fill sur « la barre 4h SUIVANTE » — si un tick a été manqué
    (redéploiement, panne), la barre reçue est PLUS TARDIVE : plus de temps pour
    traverser la limite = fill-rate gonflé, le sens d'auto-flatterie exact que le
    prereg interdit. Une entrée dont la barre suivante n'a pas été observée n'a PAS
    de mesure définie : elle est marquée VOID (comptée à part, jamais dans les
    stats ni le verdict — ce n'est pas une exclusion post-hoc, aucun outcome
    n'existe à exclure). Une barre trop VIEILLE (la barre de décision elle-même,
    double appel du même tick) laisse l'entrée en attente.

    Fail-soft by construction: never raises.
    """
    try:
        high = float(next_bar_ohlc["high"])
        low = float(next_bar_ohlc["low"])
        open_ = float(next_bar_ohlc["open"])
        entries = _load(data_dir)
        changed = False
        for e in entries:
            if e.get("symbol") != symbol or e.get("resolved"):
                continue
            dec_ts = e.get("decision_bar_ts")
            if dec_ts is not None and bar_ts is not None:
                delta = int(bar_ts) - int(dec_ts)
                if delta <= 0:
                    continue  # même barre ou plus vieille : rien à résoudre encore
                if abs(delta - BAR_INTERVAL_MS) > _NEXT_BAR_TOLERANCE_MS:
                    # barre plus tardive que LA suivante : mesure indéfinie -> VOID
                    e["resolved"] = True
                    e["void"] = True
                    changed = True
                    continue
            side = e.get("side")
            decision_close = float(e["decision_close"])
            for _off, limit_key, strict_key, touch_key in (
                (0.0, "limit_0", "fill_strict_0", "fill_touch_0"),
                (5.0, "limit_5", "fill_strict_5", "fill_touch_5"),
            ):
                limit = float(e[limit_key])
                if side == "BUY":
                    strict_fill = low < limit
                    touch_fill = low <= limit
                else:
                    strict_fill = high > limit
                    touch_fill = high >= limit
                e[strict_key] = bool(strict_fill)
                e[touch_key] = bool(touch_fill)
            if not e["fill_strict_0"]:
                if side == "BUY":
                    chase = (open_ - decision_close) / decision_close * 10_000.0
                else:
                    chase = (decision_close - open_) / decision_close * 10_000.0
                e["chase_cost_bps"] = float(chase)
            else:
                e["chase_cost_bps"] = None
            e["resolved"] = True
            changed = True
        if changed:
            _save(entries, data_dir)
    except Exception as e:  # pragma: no cover - defensive, exercised via readonly-dir test
        log.warning("shadow_maker_resolve_pending_failed", symbol=symbol, error=str(e))


def _pop_stats(pop: list[dict[str, Any]]) -> dict[str, float | None]:
    n = len(pop)
    if n == 0:
        return {
            "fill_strict_0": None, "fill_touch_0": None,
            "fill_strict_5": None, "fill_touch_5": None,
            "chase_cost_bps_mean": None, "gain_net_bps_est": None,
        }
    fs0 = sum(1 for e in pop if e.get("fill_strict_0")) / n
    ft0 = sum(1 for e in pop if e.get("fill_touch_0")) / n
    fs5 = sum(1 for e in pop if e.get("fill_strict_5")) / n
    ft5 = sum(1 for e in pop if e.get("fill_touch_5")) / n
    chases = [e["chase_cost_bps"] for e in pop if e.get("chase_cost_bps") is not None]
    chase_mean = (sum(chases) / len(chases)) if chases else 0.0
    gain_net = fs0 * GAIN_PAR_FILL_BPS - (1 - fs0) * chase_mean
    return {
        "fill_strict_0": fs0, "fill_touch_0": ft0,
        "fill_strict_5": fs5, "fill_touch_5": ft5,
        "chase_cost_bps_mean": chase_mean, "gain_net_bps_est": gain_net,
    }


def summary(*, data_dir: Path | str | None = None) -> dict[str, Any]:
    """Current shadow-maker measurement snapshot (fail-soft; see the prereg for the
    frozen definitions). ``verdict_at_40`` is None until 40 signals have resolved,
    then PASS/FAIL forever from that first-40 reading — never a second window."""
    try:
        entries = _load(data_dir)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("shadow_maker_summary_failed", error=str(e))
        entries = []

    n_logged = len(entries)
    # Les VOID (barre suivante jamais observée — écart/redéploiement) sont comptés à
    # part et n'entrent NI dans les stats NI dans le verdict : leur mesure n'est pas
    # définie par la règle gelée. Transparence obligatoire, jamais d'exclusion cachée.
    voided = [e for e in entries if e.get("resolved") and e.get("void")]
    resolved = [e for e in entries if e.get("resolved") and not e.get("void")]
    n_resolved = len(resolved)
    live = _pop_stats(resolved)

    verdict = None
    if n_resolved >= VERDICT_N:
        frozen = _pop_stats(resolved[:VERDICT_N])
        fs0 = frozen["fill_strict_0"] or 0.0
        gain_net = frozen["gain_net_bps_est"] or 0.0
        verdict = (
            "PASS"
            if fs0 >= VERDICT_FILL_STRICT_MIN and gain_net >= VERDICT_GAIN_NET_MIN_BPS
            else "FAIL"
        )

    return {
        "n_logged": n_logged,
        "n_resolved": n_resolved,
        "n_void": len(voided),
        **live,
        "verdict_at_40": verdict,
        "prereg": PREREG_PATH,
    }
