"""S2-en-largeur DURCI — univers point-in-time incluant les perps délistés
(research_breadth/PREREG_HARDENED.md), UN SEUL essai (registre crypto 1910->1911,
source `s2_breadth_hardened_prereg`, bumpé par l'orchestrateur AVANT ce calcul, commit
004529e).

EXTENSION de gate_s2_breadth.py (importé verbatim, jamais modifié) : règle S2 GELÉE
identique (funding_z fenêtre 180, z>=2.0, long-only, hold 24 barres 4h, warmup 200, coût
taker 15bps RT), mêmes routines de panier date-level dé-chevauché (fxgate.event_basket_gate
/ _dedup_basket_events), même joint-null MC 5 seeds, même OOS-by-date, mêmes fenêtres bear
(bear_windows() de gate_s2_breadth.py, appliquée sur la MÊME série BTC -> bornes de fenêtres
identiques au run original, aucune redéfinition).

Ce qui est AJOUTÉ (le durcissement, cf. PREREG_HARDENED.md) :
  1. 4 morts qualifiés point-in-time (sur 131 candidats testés : 122 SETTLING actuels de
     exchangeInfo + 9 délistings historiques plus anciens trouvés via web -- cf. rapport de
     mission pour le détail complet du screening) : MKR, BLZ, FTM, REEF. Chacun ENTRE dans
     l'univers à sa date point-in-time (30j-vol>=$200M ET historique funding>=3ans
     SIMULTANÉMENT) et SORT à sa vraie date de délisting (dernière barre à volume réel non
     nul + dernière entrée funding réelle -- PAS `deliveryDate` d'exchangeInfo, vérifié non
     fiable : ex. OMGUSDT deliveryDate=2025-01-31 mais funding réel jusqu'à 2026-06-18).
  2. Sortie forcée au délisting à coût doublé (30bps au lieu de 15bps) sur la jambe d'un
     événement DEAD dont l'horizon de hold est tronqué par la fin réelle des données de cet
     instrument (_basket_event_return_hardened ci-dessous, extension de
     fxgate._basket_event_return).
  3. Double comptabilité : univers survivants-seuls (12, = le run original, non retouché)
     vs univers durci (12+4), rapportés séparément.

AUCUN paramètre de la règle S2 n'est re-tuné ; AUCUNE fenêtre bear/OOS n'est redécoupée ;
les 12 symboles originaux gardent leur traitement intégral d'origine (pas de re-dérivation
point-in-time pour eux -- hors mandat, cf. rapport de mission : le point-in-time n'est
appliqué qu'aux 4 morts ajoutés, pour ne pas casser la comparabilité de la base
"survivants-seuls").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lacarbonnerie/Developer/hermes-hedge-fund")
sys.path.insert(0, str(ROOT / "research_breadth"))
sys.path.insert(0, str(ROOT / "research_crypto2"))
sys.path.insert(0, str(ROOT / "research_fx"))

import cryptolib  # noqa: E402
cryptolib.DATA = ROOT / "research_breadth" / "data"

import gate_s2_breadth as g0  # noqa: E402  -- le script ORIGINAL, importé tel quel (jamais modifié)
from cryptolib import resample_4h, attach_funding, span_years, COST_TAKER_RT  # noqa: E402
from fxgate import _dedup_basket_events, _sharpe  # noqa: E402

OUT = ROOT / "research_breadth"
DATA = ROOT / "research_breadth" / "data"

COST_FORCED_EXIT = 2.0 * COST_TAKER_RT  # 30bps -- "coût de sortie DOUBLÉ" (PREREG_HARDENED §2)

# Morts qualifiés point-in-time (cf. fetch_dead_universe.py pour la provenance et le
# screening complet). entry_ts_ms = première date où (a) vol30j>=$200M ET (b) historique
# funding>=3ans tiennent SIMULTANÉMENT ; delisting_ts_ms = vraie date de mort empirique
# (dernière barre à volume réel non nul, recoupée avec la dernière entrée funding réelle).
DEAD_QUALIFIED = {
    "MKR": {"entry_ts_ms": 1692057600000, "delisting_ts_ms": 1757318400000},
    "BLZ": {"entry_ts_ms": 1694995200000, "delisting_ts_ms": 1750320000003},
    "FTM": {"entry_ts_ms": 1703548800000, "delisting_ts_ms": 1750320000003},
    "REEF": {"entry_ts_ms": 1726185600000, "delisting_ts_ms": 1750320000003},
}


def load_4h_with_funding_dead(coin: str) -> pd.DataFrame:
    """Identique à cryptolib.load_4h_with_funding mais sur les fichiers _hardened des morts,
    tronqué STRICTEMENT à la vraie date de délisting (le fetch a inclus un buffer de 2j après
    la dernière barre réelle -- on l'exclut ici pour ne jamais faire tourner la règle sur du
    padding gelé volume=0)."""
    delisting_ts = DEAD_QUALIFIED[coin]["delisting_ts_ms"]
    b1h = pd.read_csv(DATA / f"{coin}_1h_hardened.csv")
    b1h = b1h[b1h["ts"] <= delisting_ts].reset_index(drop=True)
    b = resample_4h(b1h)
    f = pd.read_csv(DATA / f"{coin}_funding_hardened.csv")[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)
    f = f[f["ts"] <= delisting_ts].reset_index(drop=True)
    return attach_funding(b, f)


def long_events_dead(coin: str):
    """Verbatim g0.long_events, pointé sur le loader des morts. Puis filtré au point-in-time :
    un événement AVANT la date d'entrée du perp ne compte pas (il n'était pas encore "dans
    l'univers") ; tout événement après l'entrée compte jusqu'au bout (déjà borné par le
    troncage à delisting_ts_ms dans le loader ci-dessus)."""
    d = load_4h_with_funding_dead(coin)
    close = d["close"].to_numpy()
    ts = d["ts"].to_numpy()
    fz = g0.funding_z(d["funding_rate"].to_numpy(), g0.LIVE_WIN)
    entry_ts = DEAD_QUALIFIED[coin]["entry_ts_ms"]
    ev_all = [(i, 1) for i in range(g0.WARMUP, len(close))
              if np.isfinite(fz[i]) and fz[i] >= g0.LIVE_Z]
    ev = [(i, dr) for i, dr in ev_all if ts[i] >= entry_ts]
    n_dropped_pre_entry = len(ev_all) - len(ev)
    return d, close, ts, fz, ev, n_dropped_pre_entry


def _basket_event_return_hardened(date_ms, direction, instruments, dead_coins, horizon,
                                   cost_normal, cost_forced_exit, entry_ts_by_coin,
                                   lag_days=0):
    """Extension de fxgate._basket_event_return : identique en tout point (même look-ahead
    fix -- un instrument dont l'historique commence APRES la date de l'evenement est exclu),
    PLUS un second garde-fou point-in-time propre au durcissement : un mort dont les DONNEES
    de prix remontent a son ONBOARDING (2020-2021 pour MKR/BLZ/FTM/REEF) mais qui n'a pas
    encore satisfait (a)+(b) a la date de l'evenement (avant son `entry_ts_by_coin`) est
    EGALEMENT exclu de la jambe -- sinon un mort "existe" dans le panier des annees avant
    d'y etre reellement entre (bug trouve et corrige le 2026-07-16 : sans ce garde-fou, un
    evenement declenche par un SURVIVANT en 2021 incluait deja la jambe MKR/BLZ/FTM/REEF,
    alors qu'aucun des 4 n'entre dans l'univers avant 2023-08 au plus tot).
    PLUS le cout de sortie force : pour un instrument du set `dead_coins` dont le hold est
    TRONQUE par la fin reelle de ses propres donnees (h < horizon, i.e. sortie forcee avant
    la fin naturelle du hold 24 barres), le cout de CETTE jambe est double
    (cost_forced_exit=30bps) au lieu du cout normal (15bps) -- PREREG_HARDENED §2 "sortie
    forcee au delisting... cout de sortie double". Un survivant (BTC..TLM) ou un mort dont
    l'evenement a le temps de clore naturellement (h==horizon) paie toujours le cout normal."""
    target = int(date_ms) + int(lag_days) * 86_400_000
    rets = []
    for sym, d in instruments.items():
        close, ts, sign = d["close"], d["ts"], d["sign"]
        if target < int(ts[0]):
            continue
        if sym in dead_coins and target < entry_ts_by_coin.get(sym, 0):
            continue
        i = int(np.searchsorted(ts, target, side="left"))
        if i >= len(close) - 1:
            continue
        h = min(horizon, len(close) - 1 - i)
        if h <= 0:
            continue
        forced_exit = (sym in dead_coins) and (h < horizon)
        c = cost_forced_exit if forced_exit else cost_normal
        rets.append(float(direction) * float(sign) * (close[i + h] / close[i] - 1.0) - c)
    return float(np.mean(rets)) if rets else None


def event_basket_gate_hardened(events, instruments, dead_coins, entry_ts_by_coin, horizon,
                                cost_normal, cost_forced_exit, n_sims=5000, seed=0,
                                frac_oos=0.4, no_overlap=True):
    """Miroir de fxgate.event_basket_gate, avec _basket_event_return_hardened (garde-fou
    point-in-time + cout de sortie forcee) pour la jambe reelle ET pour le null MC
    (traitement identique aux deux, pas de double-standard -- une ancre aleatoire avant
    l'entree d'un mort ou pres de sa fin de vie est traitee exactement comme un vrai
    evenement le serait)."""
    if no_overlap and instruments:
        ref_ts = next(iter(instruments.values()))["ts"]
        events = _dedup_basket_events(events, ref_ts, horizon, 0)
    real = [
        (d, _basket_event_return_hardened(d, dr, instruments, dead_coins, horizon,
                                           cost_normal, cost_forced_exit, entry_ts_by_coin, 0))
        for d, dr in events
    ]
    real = [(d, r) for d, r in real if r is not None]
    if not real:
        return {"n_events": 0, "sleeve_mean": 0.0, "sleeve_sharpe": 0.0,
                "oos_sharpe": 0.0, "mc_p": 1.0, "per_event": []}
    dates = np.array([d for d, _ in real], dtype=float)
    rets = np.array([r for _, r in real], dtype=float)
    real_mean = float(rets.mean())

    lo, hi = dates.min(), dates.max()
    oos = _sharpe(rets[dates >= lo + (1.0 - frac_oos) * (hi - lo)]) if hi > lo else 0.0

    dirs = [dr for _, dr in events]
    all_ts = np.concatenate([d["ts"] for d in instruments.values()])
    a_lo = int(all_ts.min())
    a_hi = int(all_ts.max()) - horizon * 2 * 86_400_000
    rng = np.random.default_rng(seed)
    beats = 0
    if a_hi <= a_lo:
        return {"n_events": len(real), "sleeve_mean": real_mean, "sleeve_sharpe": _sharpe(rets),
                "oos_sharpe": oos, "mc_p": 1.0, "per_event": list(map(float, rets))}
    for _ in range(n_sims):
        anchors = rng.integers(a_lo, a_hi, size=len(dirs))
        nr = [
            _basket_event_return_hardened(int(a), dr, instruments, dead_coins, horizon,
                                           cost_normal, cost_forced_exit, entry_ts_by_coin, 0)
            for a, dr in zip(anchors, dirs)
        ]
        nr = [x for x in nr if x is not None]
        if nr and float(np.mean(nr)) >= real_mean:
            beats += 1
    return {
        "n_events": len(real), "sleeve_mean": real_mean, "sleeve_sharpe": _sharpe(rets),
        "oos_sharpe": oos, "mc_p": beats / n_sims, "per_event": list(map(float, rets)),
    }


def basket_mean_in_window_hardened(events, instruments, dead_coins, entry_ts_by_coin, horizon,
                                    cost_normal, cost_forced_exit, lo_ms, hi_ms):
    rets = []
    for d, dr in events:
        if lo_ms <= d <= hi_ms:
            r = _basket_event_return_hardened(d, dr, instruments, dead_coins, horizon,
                                               cost_normal, cost_forced_exit, entry_ts_by_coin, 0)
            if r is not None:
                rets.append(r)
    if not rets:
        return None, 0
    return float(np.mean(rets)), len(rets)


def main():
    # ---- (A) univers SURVIVANTS-SEULS : reproduction bit-exacte de gate_s2_breadth.py -------
    symbols = g0.universe_symbols()
    per_sym = g0.per_symbol_events(symbols)
    res_wide_survivors, events_survivors, instr_survivors = g0.build_basket(symbols, per_sym)
    dates_survivors = g0.dedup_dates(events_survivors, instr_survivors, g0.LIVE_H)
    span_survivors = span_years(np.concatenate([per_sym[c]["ts"] for c in symbols]))
    print(f"[A] survivants-seuls (12): n_events_deduped={res_wide_survivors['n_events']} "
          f"sleeve_mean_bps={res_wide_survivors['sleeve_mean']*1e4:.2f}")

    # ---- (B) morts qualifiés : événements point-in-time --------------------------------------
    dead_coins = list(DEAD_QUALIFIED.keys())
    entry_ts_by_coin = {c: DEAD_QUALIFIED[c]["entry_ts_ms"] for c in dead_coins}
    per_dead = {}
    dead_events_raw = {}
    n_dropped_pre_entry = {}
    for coin in dead_coins:
        d, close, ts, fz, ev, n_dropped = long_events_dead(coin)
        per_dead[coin] = {"d": d, "close": close, "ts": ts, "fz": fz, "ev": ev}
        dead_events_raw[coin] = len(ev)
        n_dropped_pre_entry[coin] = n_dropped
        print(f"[B] {coin}: {len(ev)} evenements point-in-time (entry->delisting), "
              f"{n_dropped} evenements pre-entree exclus, span_data="
              f"{pd.to_datetime(ts.min(), unit='ms').date()}->{pd.to_datetime(ts.max(), unit='ms').date()}")

    # ---- (C) univers DURCI (12 + 4) -----------------------------------------------------------
    instr_hardened = dict(instr_survivors)  # meme objets (close/ts/sign) que (A), non retouches
    for coin in dead_coins:
        instr_hardened[coin] = {"close": per_dead[coin]["close"], "ts": per_dead[coin]["ts"], "sign": 1}

    events_hardened = list(events_survivors)  # les 12 survivants gardent TOUS leurs evenements bruts
    for coin in dead_coins:
        for i, dr in per_dead[coin]["ev"]:
            events_hardened.append((int(per_dead[coin]["ts"][i]), dr))

    res_hardened = event_basket_gate_hardened(
        events_hardened, instr_hardened, dead_coins, entry_ts_by_coin, g0.LIVE_H, COST_TAKER_RT,
        COST_FORCED_EXIT, n_sims=5000, seed=0, frac_oos=0.4, no_overlap=True,
    )
    dates_hardened = sorted(int(d) for d, _ in _dedup_basket_events(
        events_hardened, next(iter(instr_hardened.values()))["ts"], g0.LIVE_H, 0))
    span_hardened = span_years(np.concatenate([instr_hardened[c]["ts"] for c in instr_hardened]))
    print(f"[C] DURCI (12+4): n_events_deduped={res_hardened['n_events']} "
          f"sleeve_mean_bps={res_hardened['sleeve_mean']*1e4:.2f}")

    # joint-null MC, 5 seeds, panier DURCI
    ps_hardened = []
    for s in range(5):
        r = event_basket_gate_hardened(events_hardened, instr_hardened, dead_coins,
                                        entry_ts_by_coin, g0.LIVE_H,
                                        COST_TAKER_RT, COST_FORCED_EXIT, n_sims=5000, seed=s,
                                        frac_oos=0.4, no_overlap=True)
        ps_hardened.append(r["mc_p"])
    mc_p_max5_hardened = float(max(ps_hardened))

    # OOS-by-date moities strictes 50/50, panier DURCI
    per_event_hardened = []
    for d in dates_hardened:
        r = _basket_event_return_hardened(d, 1, instr_hardened, dead_coins, g0.LIVE_H,
                                           COST_TAKER_RT, COST_FORCED_EXIT, entry_ts_by_coin, 0)
        if r is not None:
            per_event_hardened.append((d, r))
    de_h = np.array([d for d, _ in per_event_hardened], dtype=float)
    re_h = np.array([r for _, r in per_event_hardened], dtype=float)
    if de_h.size:
        mid_h = de_h.min() + 0.5 * (de_h.max() - de_h.min())
        oos_half_sharpe_h = _sharpe(re_h[de_h >= mid_h])
        is_half_sharpe_h = _sharpe(re_h[de_h < mid_h])
        oos_half_mean_bps_h = float(re_h[de_h >= mid_h].mean() * 1e4) if (de_h >= mid_h).any() else 0.0
        is_half_mean_bps_h = float(re_h[de_h < mid_h].mean() * 1e4) if (de_h < mid_h).any() else 0.0
        oos_half_n_h = int((de_h >= mid_h).sum())
        is_half_n_h = int((de_h < mid_h).sum())
    else:
        oos_half_sharpe_h = is_half_sharpe_h = oos_half_mean_bps_h = is_half_mean_bps_h = 0.0
        oos_half_n_h = is_half_n_h = 0

    # ---- (D) evenements marginaux : durci \ survivants-seuls -----------------------------------
    set_survivors = set(dates_survivors)
    set_hardened = set(dates_hardened)
    marginal_dates_h = sorted(set_hardened - set_survivors)
    marg_events_h = [(d, 1) for d in marginal_dates_h]
    marg_mean_h, marg_n_h = basket_mean_in_window_hardened(
        marg_events_h, instr_hardened, dead_coins, entry_ts_by_coin, g0.LIVE_H, COST_TAKER_RT,
        COST_FORCED_EXIT, -1, 10**15)
    marg_years_h = pd.to_datetime(pd.Series(marginal_dates_h, dtype="int64"), unit="ms", utc=True).dt.year
    marg_by_year_h = marg_years_h.value_counts().sort_index().to_dict() if len(marginal_dates_h) else {}
    pct_marginal_h = (len(marginal_dates_h) / res_wide_survivors["n_events"] * 100.0) if res_wide_survivors["n_events"] > 0 else float("nan")

    # ---- (E) fenetres bear -- bornes IDENTIQUES au run original (meme fonction, meme serie BTC) --
    bwins = g0.bear_windows(per_sym, coin="BTC")
    bear_results_hardened = {}
    for name, (lo, hi) in bwins.items():
        m, n = basket_mean_in_window_hardened(events_hardened, instr_hardened, dead_coins,
                                               entry_ts_by_coin, g0.LIVE_H, COST_TAKER_RT,
                                               COST_FORCED_EXIT, lo, hi)
        bear_results_hardened[name] = {
            "lo_date": str(pd.to_datetime(lo, unit="ms", utc=True).date()),
            "hi_date": str(pd.to_datetime(hi, unit="ms", utc=True).date()),
            "mean_net_bps": (m * 1e4) if m is not None else None,
            "n_events": n,
        }
    bear_all_positive_hardened = all(
        (v["mean_net_bps"] is None) or (v["mean_net_bps"] > 0.0) for v in bear_results_hardened.values()
    )

    # fenetres bear sur SURVIVANTS-SEULS (recalculees ici avec la MEME fonction hardened mais
    # dead_coins=[] -- doit reproduire bit-a-bit gate_s2_breadth.py/JOURNAL.md §8.3, verification
    # croisee de l'instrument)
    bear_results_survivors = {}
    for name, (lo, hi) in bwins.items():
        m, n = basket_mean_in_window_hardened(events_survivors, instr_survivors, [], {},
                                               g0.LIVE_H, COST_TAKER_RT, COST_FORCED_EXIT, lo, hi)
        bear_results_survivors[name] = {"mean_net_bps": (m * 1e4) if m is not None else None, "n_events": n}

    # ---- (F) cecite 2022 : evenements bruts (point-in-time, pre-dedup) sur le panier DURCI ------
    n_2022_survivors = 0
    for c in symbols:
        ts_c = per_sym[c]["ts"]
        for i, _ in per_sym[c]["ev"]:
            dt = pd.Timestamp(int(ts_c[i]), unit="ms", tz="UTC")
            if dt.year == 2022:
                n_2022_survivors += 1
    n_2022_dead = 0
    for coin in dead_coins:
        ts_c = per_dead[coin]["ts"]
        for i, _ in per_dead[coin]["ev"]:
            dt = pd.Timestamp(int(ts_c[i]), unit="ms", tz="UTC")
            if dt.year == 2022:
                n_2022_dead += 1
    n_2022_hardened_total = n_2022_survivors + n_2022_dead

    # ---- (G) DSR contre le N reel du registre (courant, verifie via governance/) ----------------
    n_trials_registry = json.loads((ROOT / "governance" / "trials_by_book.json").read_text())["counts"]["crypto"]
    sys.path.insert(0, str(ROOT / "src"))
    from hermes.learning.stats import deflated_sharpe_ratio  # noqa: E402
    from fxgate import realize_event_returns  # noqa: E402

    per_asset_sharpes = []
    for c in symbols:
        tr = realize_event_returns(per_sym[c]["close"], per_sym[c]["ts"], per_sym[c]["ev"],
                                    g0.LIVE_H, COST_TAKER_RT, warmup=g0.WARMUP)
        r = np.array([t["ret_net"] for t in tr], dtype=float)
        per_asset_sharpes.append(_sharpe(r))
    for coin in dead_coins:
        tr = realize_event_returns(per_dead[coin]["close"], per_dead[coin]["ts"], per_dead[coin]["ev"],
                                    g0.LIVE_H, COST_TAKER_RT, warmup=0)  # warmup deja applique via WARMUP=200 sur bar-index dans long_events_dead
        r = np.array([t["ret_net"] for t in tr], dtype=float)
        per_asset_sharpes.append(_sharpe(r))
    sr_var_hardened = float(np.var(np.array(per_asset_sharpes + [res_wide_survivors["sleeve_sharpe"], res_hardened["sleeve_sharpe"]]), ddof=1))
    dsr_hardened = float(deflated_sharpe_ratio(re_h.tolist(), sr_var_hardened, n_trials_registry)) if re_h.size else 0.0

    # ---- verdict MECANIQUE (chiffres seulement -- le verdict lettre-PREREG est tranche par
    # l'orchestrateur, cf. mandat ; on rapporte quand meme les 4 checks pour tracabilite) --------
    fail_reasons = []
    if not (pct_marginal_h >= 50.0 if not np.isnan(pct_marginal_h) else False):
        fail_reasons.append(f"marginaux durci={len(marginal_dates_h)} (+{pct_marginal_h:.1f}%) < seuil +50%")
    if not (mc_p_max5_hardened < 0.05):
        fail_reasons.append(f"joint-null durci max(5 seeds)={mc_p_max5_hardened:.4f} >= 0.05")
    if not (oos_half_mean_bps_h > 0.0):
        fail_reasons.append(f"OOS-by-date durci negatif: {oos_half_mean_bps_h:.2f} bps")
    if not bear_all_positive_hardened:
        neg = {k: v["mean_net_bps"] for k, v in bear_results_hardened.items()
               if v["mean_net_bps"] is not None and v["mean_net_bps"] <= 0.0}
        fail_reasons.append(f"bear-window durci ECHOUE: {neg}")

    results = {
        "prereg_source": "s2_breadth_hardened_prereg",
        "registry_n_at_calc": n_trials_registry,
        "dead_qualified_coins": dead_coins,
        "dead_screening_summary": {
            "n_candidates_settling_exchangeinfo": 122,
            "n_candidates_historical_pre_settling_tested_via_web": 9,
            "n_candidates_total_tested": 131,
            "n_qualified_point_in_time": 4,
        },
        "dead_entry_exit": {c: {
            "entry_date": str(pd.to_datetime(DEAD_QUALIFIED[c]["entry_ts_ms"], unit="ms").date()),
            "delisting_date": str(pd.to_datetime(DEAD_QUALIFIED[c]["delisting_ts_ms"], unit="ms").date()),
            "n_events_point_in_time": dead_events_raw[c],
            "n_events_dropped_pre_entry": n_dropped_pre_entry[c],
        } for c in dead_coins},
        "basket_survivors_only_12": {
            "n_events_deduped": res_wide_survivors["n_events"],
            "span_years": span_survivors,
            "n_events_per_year": res_wide_survivors["n_events"] / span_survivors if span_survivors > 0 else 0.0,
            "sleeve_mean_net_bps": res_wide_survivors["sleeve_mean"] * 1e4,
            "sleeve_sharpe": res_wide_survivors["sleeve_sharpe"],
            "oos_sharpe_40pct": res_wide_survivors["oos_sharpe"],
            "mc_p_seed0": res_wide_survivors["mc_p"],
        },
        "basket_hardened_12plus4": {
            "n_events_deduped": res_hardened["n_events"],
            "span_years": span_hardened,
            "n_events_per_year": res_hardened["n_events"] / span_hardened if span_hardened > 0 else 0.0,
            "sleeve_mean_net_bps": res_hardened["sleeve_mean"] * 1e4,
            "sleeve_sharpe": res_hardened["sleeve_sharpe"],
            "oos_sharpe_40pct": res_hardened["oos_sharpe"],
            "mc_p_seed0": res_hardened["mc_p"],
            "mc_p_5seeds": ps_hardened,
            "mc_p_max_5seeds": mc_p_max5_hardened,
            "mc_p_stable_lt05": bool(mc_p_max5_hardened < 0.05),
        },
        "marginal_events_hardened_vs_survivors": {
            "n_marginal_dates": len(marginal_dates_h),
            "pct_increase_vs_survivors_only": pct_marginal_h,
            "mean_net_bps": (marg_mean_h * 1e4) if marg_mean_h is not None else None,
            "n_used_for_mean": marg_n_h,
            "by_year": {str(k): int(v) for k, v in marg_by_year_h.items()},
        },
        "oos_by_date_halves_hardened_basket": {
            "in_sample_first_half": {"mean_net_bps": is_half_mean_bps_h, "sharpe": is_half_sharpe_h, "n": is_half_n_h},
            "out_of_sample_second_half": {"mean_net_bps": oos_half_mean_bps_h, "sharpe": oos_half_sharpe_h, "n": oos_half_n_h},
        },
        "bear_window_check_hardened": bear_results_hardened,
        "bear_window_check_survivors_only_crosscheck": bear_results_survivors,
        "bear_all_positive_hardened": bear_all_positive_hardened,
        "blindness_2022": {
            "n_events_2022_survivors_only_raw": n_2022_survivors,
            "n_events_2022_dead_raw": n_2022_dead,
            "n_events_2022_hardened_total_raw": n_2022_hardened_total,
            "resolved": bool(n_2022_hardened_total > 0),
            "note": "tous les morts qualifies ont une date d'entree point-in-time en 2023 ou "
                    "2024 (aucun n'atteint le seuil vol30j+funding3ans avant 2022) -- la "
                    "cecite 2022 ne PEUT PAS etre resolue par ce durcissement, par construction.",
        },
        "dsr_hardened_at_registry_n": dsr_hardened,
        "sr_variance_proxy_hardened": sr_var_hardened,
        "cost_forced_exit_bps": COST_FORCED_EXIT * 1e4,
        "mechanical_checks_hardened": {
            "marginal_gte_50pct": bool(pct_marginal_h >= 50.0) if not np.isnan(pct_marginal_h) else False,
            "joint_null_lt_05": bool(mc_p_max5_hardened < 0.05),
            "oos_positive": bool(oos_half_mean_bps_h > 0.0),
            "bear_all_positive": bear_all_positive_hardened,
            "fail_reasons": fail_reasons,
        },
    }

    (OUT / "_breadth_hardened_results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\n" + json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
