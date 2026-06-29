"""Offline tests for the central-bank-speeches corpus lake client.

CORPUS COLLECTION ONLY — no scoring/tone/z-scores. We verify the pure transforms:
  * ``visible_text`` HTML->text extraction (ported from the FOMC builder),
  * the BIS bulk unzip+normalize (in-memory zip -> common schema),
  * the BIS free-text author -> institution best-effort parse,
  * the Fed FOMC CSV -> common-schema normalizer (STATEMENTS, kind column),
  * the ECB bulk-CSV normalizer,
  * a fake-transport happy path through ``_fetch_zip_bytes`` (no live network).

NO live network anywhere. NO 122MB download.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from hermes.datalake.sources.cbspeak import (
    COMMON_SCHEMA,
    bis_unzip_to_frame,
    ecb_csv_to_frame,
    fed_fomc_to_frame,
    fetch_zip_bytes,
    normalize_bis_frame,
    parse_institution,
    visible_text,
)


# --------------------------------------------------------------------------- #
# visible_text
# --------------------------------------------------------------------------- #
def test_visible_text_strips_tags_scripts_styles_and_unescapes():
    html = (
        b"<html><head><style>.x{color:red}</style>"
        b"<script>var a = 1 < 2;</script></head>"
        b"<body><p>Inflation &amp; growth</p>"
        b"<div>remain   elevated</div></body></html>"
    )
    out = visible_text(html)
    assert out == "Inflation & growth remain elevated"
    assert "var a" not in out  # script body removed
    assert "color:red" not in out  # style body removed
    assert "<" not in out and ">" not in out


def test_visible_text_collapses_whitespace_and_handles_empty():
    assert visible_text(b"") == ""
    assert visible_text(b"<p>a</p>\n\n\t<p>b</p>") == "a b"


# --------------------------------------------------------------------------- #
# parse_institution (BIS free-text author)
# --------------------------------------------------------------------------- #
def test_parse_institution_best_effort():
    # "speaker, role, institution" free-text forms.
    assert (
        parse_institution("Christine Lagarde, President of the European Central Bank")
        == "European Central Bank"
    )
    assert (
        parse_institution("Jerome Powell, Chair of the Board of Governors of the Federal Reserve System")
        == "Federal Reserve System"
    )
    # No recognizable institution -> empty string (per spec).
    assert parse_institution("Some Speaker") == ""
    assert parse_institution("") == ""
    assert parse_institution(None) == ""


# --------------------------------------------------------------------------- #
# BIS normalize (pure, frame in -> common schema out)
# --------------------------------------------------------------------------- #
def _raw_bis_frame() -> pd.DataFrame:
    # Mirror the BIS bulk CSV's real-ish columns (best-effort tolerant mapping).
    return pd.DataFrame(
        {
            "date": ["2020-03-15", "1996-01-02"],
            "author": [
                "Christine Lagarde, President of the European Central Bank",
                "Alan Greenspan, Chairman of the Federal Reserve System",
            ],
            "title": ["Monetary policy in a pandemic", "Outlook"],
            "url": ["https://www.bis.org/review/r200315a.htm", "https://www.bis.org/r9601.htm"],
            "text": ["Body one.", "Body two."],
        }
    )


def test_normalize_bis_frame_yields_common_schema():
    out = normalize_bis_frame(_raw_bis_frame())
    assert list(out.columns) == COMMON_SCHEMA
    assert (out["source_bank"] == "BIS").all()
    assert (out["kind"] == "speech").all()
    # institution best-effort parsed from the free-text author
    assert out.iloc[0]["institution"] == "European Central Bank"
    assert out.iloc[0]["speaker"].startswith("Christine Lagarde")
    # date normalized to ISO yyyy-mm-dd string
    assert out.iloc[0]["date"] == "2020-03-15"
    assert out.iloc[1]["date"] == "1996-01-02"


def test_normalize_bis_frame_tolerates_missing_columns():
    raw = pd.DataFrame({"date": ["2001-09-11"], "author": ["X"], "text": ["t"]})
    out = normalize_bis_frame(raw)
    assert list(out.columns) == COMMON_SCHEMA
    assert out.iloc[0]["title"] == ""  # missing -> filled empty
    assert out.iloc[0]["url"] == ""
    assert out.iloc[0]["institution"] == ""  # unparseable author


# --------------------------------------------------------------------------- #
# BIS in-memory zip unzip+normalize (no live download)
# --------------------------------------------------------------------------- #
def _make_zip(csv_name: str, df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, df.to_csv(index=False))
    return buf.getvalue()


def test_bis_unzip_to_frame_reads_contained_csv_and_normalizes():
    zip_bytes = _make_zip("speeches.csv", _raw_bis_frame())
    out = bis_unzip_to_frame(zip_bytes)
    assert list(out.columns) == COMMON_SCHEMA
    assert len(out) == 2
    assert (out["source_bank"] == "BIS").all()


def test_bis_unzip_picks_the_csv_among_other_members():
    df = _raw_bis_frame()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "ignore me")
        zf.writestr("full_dataset/speeches_all.csv", df.to_csv(index=False))
    out = bis_unzip_to_frame(buf.getvalue())
    assert len(out) == 2
    assert list(out.columns) == COMMON_SCHEMA


# --------------------------------------------------------------------------- #
# fetch_zip_bytes via injected fake transport (no live network)
# --------------------------------------------------------------------------- #
def test_fetch_zip_bytes_uses_injected_client_and_browser_ua():
    captured = {}

    class _Resp:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **k):
            captured["headers"] = k.get("headers", {})

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **k):
            captured["url"] = url
            return _Resp(b"PKZIPBYTES")

    raw = fetch_zip_bytes("https://www.bis.org/speeches/speeches.zip", client_factory=_FakeClient)
    assert raw == b"PKZIPBYTES"
    assert captured["url"].endswith("speeches.zip")
    ua = captured["headers"].get("User-Agent", "")
    assert "Mozilla" in ua  # browser User-Agent required by BIS


# --------------------------------------------------------------------------- #
# Fed FOMC CSV -> common schema (STATEMENTS, kind column)
# --------------------------------------------------------------------------- #
def _fomc_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2000-02-02", "2008-12-16"],
            "release_ts": [949518000000, 1229450400000],
            "text": ["The Federal Open Market Committee voted today...", "target range ..."],
            "n_words": [197, 150],
            "url": [
                "https://www.federalreserve.gov/boarddocs/press/general/2000/20000202/default.htm",
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20081216b.htm",
            ],
            "era": ["boarddocs_general", "modern"],
        }
    )


def test_fed_fomc_to_frame_normalizes_to_common_schema_as_statements():
    out = fed_fomc_to_frame(_fomc_rows())
    assert list(out.columns) == COMMON_SCHEMA
    assert (out["speaker"] == "FOMC").all()
    assert (out["title"] == "FOMC statement").all()
    assert (out["institution"] == "Federal Reserve").all()
    assert (out["source_bank"] == "Fed").all()
    assert (out["kind"] == "statement").all()  # labeled STATEMENTS, not speeches
    assert out.iloc[0]["text"].startswith("The Federal Open Market Committee")
    assert out.iloc[0]["date"] == "2000-02-02"
    # original url is preserved
    assert "20000202" in out.iloc[0]["url"]


def test_fed_fomc_to_frame_empty_in_empty_out():
    out = fed_fomc_to_frame(pd.DataFrame(columns=["date", "text", "url"]))
    assert list(out.columns) == COMMON_SCHEMA
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# ECB bulk CSV -> common schema
# --------------------------------------------------------------------------- #
def test_ecb_csv_to_frame_normalizes():
    # The ECB all_ECB_speeches.csv is pipe-delimited: date|speakers|title|subtitle|contents
    raw = pd.DataFrame(
        {
            "date": ["2021-07-08", "1999-01-07"],
            "speakers": ["Christine Lagarde", "Wim Duisenberg"],
            "title": ["Strategy review", "Introductory statement"],
            "subtitle": ["", ""],
            "contents": ["Body A", "Body B"],
        }
    )
    out = ecb_csv_to_frame(raw)
    assert list(out.columns) == COMMON_SCHEMA
    assert (out["source_bank"] == "ECB").all()
    assert (out["institution"] == "European Central Bank").all()
    assert (out["kind"] == "speech").all()
    assert out.iloc[0]["speaker"] == "Christine Lagarde"
    assert out.iloc[0]["text"] == "Body A"
    assert out.iloc[0]["date"] == "2021-07-08"


def test_ecb_csv_to_frame_missing_contents_column_is_tolerated():
    raw = pd.DataFrame({"date": ["2020-01-01"], "speakers": ["X"], "title": ["T"]})
    out = ecb_csv_to_frame(raw)
    assert list(out.columns) == COMMON_SCHEMA
    assert out.iloc[0]["text"] == ""
