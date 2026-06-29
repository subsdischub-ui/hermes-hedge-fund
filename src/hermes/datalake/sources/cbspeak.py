"""Central-bank-speeches corpus lake client (CORPUS COLLECTION ONLY).

Builds a frozen, point-in-time text corpus of central-bank communication under
``raw_root/speeches/<bank>/`` as Parquet. NO scoring / tone / z-scores live here
(any hawk/dove demo from the research scouts is intentionally stripped). The
downstream research stream constructs signals; this module only acquires text.

Sources
-------
(1) BIS bulk (primary, verified): ``https://www.bis.org/speeches/speeches.zip``
    (~122MB, full text since 1996) fetched with a BROWSER User-Agent, unzipped
    in-memory (stdlib ``zipfile``), the contained CSV read with pandas, and
    normalized to the common schema. Per-year increments ``speeches_YYYY.zip``
    are supported by the same code path. source_bank='BIS', kind='speech'.
    BIS 'author' is a free-text "speaker, role, institution" string -> we make a
    best-effort parse of the institution (else institution='').

(2) Fed FOMC statements: the existing
    ``data/fx/fomc_statements_full.csv`` (219 rows; [date, release_ts, text,
    n_words, url, era]) is FROZEN into ``raw_root/speeches/fed/fomc_statements.parquet``
    normalized to the common schema (speaker='FOMC', title='FOMC statement',
    institution='Federal Reserve', source_bank='Fed'); labeled STATEMENTS via the
    ``kind`` column (kind='statement').

(3) ECB bulk CSV ``all_ECB_speeches.csv`` and (4) BoE listing: the URLs were NOT
    verified offline. The code is written best-effort; when a URL cannot be
    confirmed it fails SOFT (records ``manifest.blocker(...)``) rather than
    crashing the run. source_bank ECB / BoE.

Common schema (column order is load-bearing — the orchestrator unions on it)::

    [date, speaker, title, url, text, institution, source_bank, kind]

``date`` is an ISO ``YYYY-MM-DD`` string (a corpus, not a time-series; per the
shared contract, non-time-series sources document a sensible date column — here
that is ``date``). ``kind`` is 'speech' or 'statement'.

Transport is injected (``client_factory`` / ``read_csv``) so tests fake it; the
public ``run()`` does live network and must NOT be executed by tests.
"""

from __future__ import annotations

import html
import io
import os
import pathlib
import re
import zipfile
from collections.abc import Callable
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, cached_pull

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
COMMON_SCHEMA: list[str] = [
    "date",
    "speaker",
    "title",
    "url",
    "text",
    "institution",
    "source_bank",
    "kind",
]

# Browser User-Agent — BIS serves the bulk zip only to UA strings that look like
# a real browser (a bare python-requests UA is refused).
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Endpoints. (1) and (2) verified; (3)/(4) best-effort (unverified offline).
BIS_SPEECHES_ZIP = "https://www.bis.org/speeches/speeches.zip"
BIS_YEAR_ZIP = "https://www.bis.org/speeches/speeches_{year}.zip"  # per-year increment
ECB_SPEECHES_CSV = "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"  # UNVERIFIED
BOE_SPEECHES_INDEX = "https://www.bankofengland.co.uk/speeches"  # UNVERIFIED (listing page)

FED_FOMC_CSV = pathlib.Path(
    "/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/fx/fomc_statements_full.csv"
)

# Known central-bank institution names, longest-first so multi-word names win.
_KNOWN_INSTITUTIONS = sorted(
    [
        "European Central Bank",
        "Federal Reserve System",
        "Federal Reserve Board",
        "Federal Reserve Bank",
        "Federal Reserve",
        "Bank of England",
        "Bank of Japan",
        "Bank of Canada",
        "Bank of France",
        "Bank of Italy",
        "Bank of Spain",
        "Bank of Korea",
        "Bank for International Settlements",
        "Deutsche Bundesbank",
        "Bundesbank",
        "Swiss National Bank",
        "Reserve Bank of Australia",
        "Reserve Bank of New Zealand",
        "Reserve Bank of India",
        "People's Bank of China",
        "Sveriges Riksbank",
        "Riksbank",
        "Norges Bank",
        "Bank of Mexico",
        "Banco de Mexico",
        "South African Reserve Bank",
        "Monetary Authority of Singapore",
        "Hong Kong Monetary Authority",
        "National Bank of Belgium",
        "Central Bank of Ireland",
    ],
    key=len,
    reverse=True,
)


# --------------------------------------------------------------------------- #
# PURE TRANSFORMS (unit-tested offline)
# --------------------------------------------------------------------------- #
def visible_text(raw: bytes | str) -> str:
    """HTML -> visible text (ported from research_fx/build_fomc_corpus.py).

    Drops <script>/<style> bodies and all tags, unescapes HTML entities, and
    collapses runs of whitespace to single spaces. Accepts bytes or str.
    """
    t = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    t = re.sub(r"(?s)<script.*?</script>", " ", t)
    t = re.sub(r"(?s)<style.*?</style>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return html.unescape(re.sub(r"\s+", " ", t)).strip()


def parse_institution(author: Any) -> str:
    """Best-effort institution from a BIS free-text "speaker, role, institution".

    Returns the longest known central-bank name that appears in the string, or
    '' if none is recognized (per spec — leave institution empty rather than guess).
    """
    if author is None:
        return ""
    s = str(author)
    if not s.strip():
        return ""
    for inst in _KNOWN_INSTITUTIONS:
        if inst.lower() in s.lower():
            return inst
    return ""


def _iso_date(value: Any) -> str:
    """Coerce a date-ish value to an ISO ``YYYY-MM-DD`` string; '' if unparseable."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """First matching column (case-insensitive) as a string Series, else empty
    strings indexed like ``df``."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _assemble(
    df_len_index: pd.Index,
    *,
    date: pd.Series,
    speaker: pd.Series,
    title: pd.Series,
    url: pd.Series,
    text: pd.Series,
    institution: pd.Series,
    source_bank: str,
    kind: str,
) -> pd.DataFrame:
    """Build a common-schema frame from per-column Series (already aligned)."""
    out = pd.DataFrame(
        {
            "date": [_iso_date(v) for v in date],
            "speaker": list(speaker),
            "title": list(title),
            "url": list(url),
            "text": list(text),
            "institution": list(institution),
            "source_bank": source_bank,
            "kind": kind,
        },
        index=df_len_index,
    )
    return out[COMMON_SCHEMA].reset_index(drop=True)


def normalize_bis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """BIS bulk CSV frame -> common schema. Tolerant of column-name variants.

    BIS 'author' is a free-text "speaker, role, institution" -> we keep the whole
    string as ``speaker`` and best-effort parse ``institution`` out of it.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COMMON_SCHEMA)
    author = _col(raw, "author", "speaker", "speakers")
    return _assemble(
        raw.index,
        date=_col(raw, "date", "speech_date", "Date"),
        speaker=author,
        title=_col(raw, "title", "Title"),
        url=_col(raw, "url", "link", "URL"),
        text=_col(raw, "text", "speech", "content", "contents", "body"),
        institution=author.map(parse_institution),
        source_bank="BIS",
        kind="speech",
    )


def bis_unzip_to_frame(zip_bytes: bytes) -> pd.DataFrame:
    """Unzip BIS bulk bytes IN-MEMORY, read the contained CSV with pandas, and
    normalize to the common schema. Picks the (largest) ``.csv`` member, so it
    works whether the archive is a bare CSV or wraps it in a folder alongside
    READMEs."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError("no .csv member in BIS zip")
        # Largest CSV is the speeches table (a README csv, if any, is tiny).
        member = max(csv_members, key=lambda n: zf.getinfo(n).file_size)
        with zf.open(member) as fh:
            raw = pd.read_csv(fh)
    return normalize_bis_frame(raw)


def fed_fomc_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """FOMC statements CSV ([date, release_ts, text, n_words, url, era]) ->
    common schema. Labeled STATEMENTS via kind='statement'."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COMMON_SCHEMA)
    n = len(raw)
    return _assemble(
        raw.index,
        date=_col(raw, "date"),
        speaker=pd.Series(["FOMC"] * n, index=raw.index),
        title=pd.Series(["FOMC statement"] * n, index=raw.index),
        url=_col(raw, "url"),
        text=_col(raw, "text"),
        institution=pd.Series(["Federal Reserve"] * n, index=raw.index),
        source_bank="Fed",
        kind="statement",
    )


def ecb_csv_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """ECB all_ECB_speeches.csv (date|speakers|title|subtitle|contents) ->
    common schema. institution fixed to 'European Central Bank'."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COMMON_SCHEMA)
    n = len(raw)
    return _assemble(
        raw.index,
        date=_col(raw, "date"),
        speaker=_col(raw, "speakers", "speaker", "author"),
        title=_col(raw, "title"),
        url=_col(raw, "url", "link"),
        text=_col(raw, "contents", "content", "text", "speech"),
        institution=pd.Series(["European Central Bank"] * n, index=raw.index),
        source_bank="ECB",
        kind="speech",
    )


# --------------------------------------------------------------------------- #
# TRANSPORT (injectable; default constructs a real client)
# --------------------------------------------------------------------------- #
def fetch_zip_bytes(
    url: str,
    *,
    client_factory: Callable[..., Any] | None = None,
    timeout: float = 300.0,
) -> bytes:
    """GET ``url`` with a browser User-Agent and return the raw body bytes.

    ``client_factory`` is injectable (a context-manager client exposing
    ``.get(url) -> resp`` with ``.content`` and ``.raise_for_status()``); default
    constructs a real ``httpx.Client``. This is the ONE function that touches the
    network — tests inject a fake factory so no live call happens.
    """
    if client_factory is None:
        import httpx

        client_factory = httpx.Client
    headers = {"User-Agent": BROWSER_UA, "Accept": "*/*"}
    with client_factory(headers=headers, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


# --------------------------------------------------------------------------- #
# PUBLIC ENTRYPOINT — live network. DO NOT execute from tests.
# --------------------------------------------------------------------------- #
def run(raw_root: pathlib.Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull the central-bank-speeches corpus into ``raw_root/speeches/<bank>/``.

    Each source is wrapped so a single failure becomes a ``manifest.blocker(...)``
    and the rest of the run continues (fail-soft). ``cached_pull`` already
    fail-softs fetch-time errors; here we additionally guard setup/validation so
    an unverified-URL source degrades to a blocker rather than crashing.

    Reads no API keys (these feeds are unauthenticated), but the orchestrator's
    ``os.environ`` is honored at call time should an override be supplied:
      * ``HERMES_BIS_SPEECHES_URL`` — override the BIS bulk-zip URL.
      * ``HERMES_ECB_SPEECHES_CSV_URL`` — override the (unverified) ECB CSV URL.
    """
    raw_root = pathlib.Path(raw_root)
    speeches = raw_root / "speeches"

    # -- (2) Fed FOMC statements: FREEZE the existing CSV (no network) ---------
    fed_target = speeches / "fed" / "fomc_statements.parquet"

    def _fetch_fed() -> pd.DataFrame | None:
        if not FED_FOMC_CSV.exists():
            return None
        return fed_fomc_to_frame(pd.read_csv(FED_FOMC_CSV))

    try:
        cached_pull(
            fed_target,
            _fetch_fed,
            source="cbspeak",
            endpoint="fed/fomc_statements",
            params={"src": str(FED_FOMC_CSV)},
            manifest=manifest,
        )
        if not FED_FOMC_CSV.exists():
            manifest.blocker(
                source="cbspeak",
                endpoint="fed/fomc_statements",
                params={"src": str(FED_FOMC_CSV)},
                error=f"FOMC source CSV not found: {FED_FOMC_CSV}",
                action="run research_fx/build_fomc_corpus.py to (re)build fomc_statements_full.csv, "
                "or point FED_FOMC_CSV at the frozen copy",
            )
    except Exception as e:  # belt-and-braces; cached_pull already fail-softs fetch
        manifest.blocker(
            source="cbspeak", endpoint="fed/fomc_statements",
            params={"src": str(FED_FOMC_CSV)}, error=repr(e),
            action="inspect the FOMC freeze step",
        )

    # -- (1) BIS bulk speeches.zip (PRIMARY, verified) -------------------------
    bis_url = os.environ.get("HERMES_BIS_SPEECHES_URL", BIS_SPEECHES_ZIP)
    bis_target = speeches / "bis" / "speeches_all.parquet"

    def _fetch_bis() -> pd.DataFrame | None:
        return bis_unzip_to_frame(fetch_zip_bytes(bis_url))

    try:
        cached_pull(
            bis_target,
            _fetch_bis,
            source="cbspeak",
            endpoint="bis/speeches_all",
            params={"url": bis_url},
            manifest=manifest,
        )
    except Exception as e:
        manifest.blocker(
            source="cbspeak", endpoint="bis/speeches_all",
            params={"url": bis_url}, error=repr(e),
            action="check BIS bulk URL / browser User-Agent / zip layout",
        )

    # -- (3) ECB bulk CSV (URL UNVERIFIED -> fail-soft) ------------------------
    ecb_url = os.environ.get("HERMES_ECB_SPEECHES_CSV_URL", ECB_SPEECHES_CSV)
    ecb_target = speeches / "ecb" / "speeches_all.parquet"

    def _fetch_ecb() -> pd.DataFrame | None:
        import httpx

        headers = {"User-Agent": BROWSER_UA, "Accept": "*/*"}
        with httpx.Client(headers=headers, timeout=120.0, follow_redirects=True) as client:
            resp = client.get(ecb_url)
            resp.raise_for_status()
            # all_ECB_speeches.csv is pipe-delimited.
            raw = pd.read_csv(io.BytesIO(resp.content), sep="|")
        return ecb_csv_to_frame(raw)

    try:
        cached_pull(
            ecb_target,
            _fetch_ecb,
            source="cbspeak",
            endpoint="ecb/speeches_all",
            params={"url": ecb_url},
            manifest=manifest,
        )
    except Exception as e:
        manifest.blocker(
            source="cbspeak", endpoint="ecb/speeches_all",
            params={"url": ecb_url}, error=repr(e),
            action="VERIFY the ECB bulk-CSV URL (all_ECB_speeches.csv) before relying on this pull",
        )

    # -- (4) BoE listing (URL UNVERIFIED -> declared blocker, no crash) --------
    # The BoE has no single bulk file; speeches are an HTML listing requiring
    # pagination + per-page visible_text extraction. The listing URL/markup were
    # not verified offline, so we DECLARE the blocker rather than guess a scraper
    # that would silently mis-parse. (visible_text() above is ready to extract
    # each speech body once the listing contract is confirmed.)
    manifest.blocker(
        source="cbspeak",
        endpoint="boe/speeches",
        params={"index": BOE_SPEECHES_INDEX},
        error="BoE speeches listing URL/markup NOT verified offline; no bulk file exists",
        action="VERIFY https://www.bankofengland.co.uk/speeches pagination + per-speech page "
        "markup, then add an enumerate-index-then-fetch loop (reuse visible_text); "
        "until then BoE contributes no rows",
    )
