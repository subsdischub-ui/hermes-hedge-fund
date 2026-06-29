"""FORWARD — the unified collection cron (one idempotent, fail-soft job, many flux).

The runner must: write each collector's snapshot through the idempotent log; keep
going when one flux fails (fail-soft, recording the blocker); and be safe to re-run
on the same UTC day (no duplicate rows). The XS reversion shadow is gated OFF by
default (fragile in-sample — see the sanity gate); funding-squeeze always runs.
"""

from __future__ import annotations

import pandas as pd

from hermes.forward import logio, run

_DAY_MS = 86_400_000
_NOW = 100 * _DAY_MS + 3_600_000


def _ok_collector():
    def c():
        df = pd.DataFrame({"ts": [_NOW], "k": ["A"], "v": [1.0]})
        return {"logs": [("flux_ok", df, ("k",), "ts")], "blockers": {}}

    c.flux_name = "ok"
    return c


def _raising_collector():
    def c():
        raise RuntimeError("feed down")

    c.flux_name = "boom"
    return c


def test_runner_writes_logs_and_is_fail_soft(tmp_path):
    manifest = run.run_forward(
        data_dir=tmp_path, now_ms=_NOW, collectors=[_ok_collector(), _raising_collector()]
    )
    assert manifest["flux"]["ok"]["status"] == "ok"
    assert manifest["flux"]["ok"]["rows_written"] == 1
    assert manifest["flux"]["boom"]["status"] == "error"  # one failure never aborts the run
    back = logio.read_log("flux_ok", data_dir=tmp_path)
    assert list(back["v"]) == [1.0]


def test_runner_is_idempotent_same_day(tmp_path):
    run.run_forward(data_dir=tmp_path, now_ms=_NOW, collectors=[_ok_collector()])
    run.run_forward(data_dir=tmp_path, now_ms=_NOW + 60_000, collectors=[_ok_collector()])
    back = logio.read_log("flux_ok", data_dir=tmp_path)
    assert len(back) == 1  # re-running the same UTC day overwrites, never duplicates


def test_default_collectors_gate_xs_reversion_off_by_default():
    names_off = run.default_collector_names(wire_xs_reversion=False)
    names_on = run.default_collector_names(wire_xs_reversion=True)
    assert "funding_squeeze_shadow" in names_off  # the live edge always accrues
    assert "xs_reversion_shadow" not in names_off  # fragile → not wired by default
    assert "xs_reversion_shadow" in names_on  # but activation is a one-flag change
    for flux in ("skew", "comex", "macro"):
        assert flux in names_off
