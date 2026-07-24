"""EDGAR adapter: locate 10-K filings and extract the Item 1A "Risk Factors"
section.

The network functions require outbound access to sec.gov / data.sec.gov and are
blocked in sandboxes whose egress is limited to package registries -- run them
where EDGAR is reachable. SEC requires a descriptive User-Agent; set a real
contact address in `USER_AGENT`.

`strip_html` and `extract_risk_factors` are pure and unit-tested offline -- they
are the part that actually feeds the signal, and the part most likely to need
tuning per filer formatting.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Optional

USER_AGENT = "research-harness contact@example.com"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(document: str) -> str:
    """Remove tags and collapse whitespace to plain text."""
    return _WS.sub(" ", _TAG.sub(" ", document)).strip()


def extract_risk_factors(document: str) -> str:
    """Best-effort slice of the Item 1A..Item 1B (Risk Factors) section.

    Works on raw HTML or already-stripped text. Returns "" if Item 1A is not
    found. Filer formatting varies wildly; treat this as a starting heuristic to
    refine against real documents, not a finished parser.
    """
    text = strip_html(document) if "<" in document else _WS.sub(" ", document).strip()
    low = text.lower()
    start = low.find("item 1a")
    if start == -1:
        return ""
    end = low.find("item 1b", start + len("item 1a"))
    if end == -1:
        end = low.find("item 2", start + len("item 1a"))
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_cik(ticker: str) -> Optional[str]:
    """Map a ticker to its zero-padded 10-digit CIK via EDGAR's ticker file."""
    data = _get_json("https://www.sec.gov/files/company_tickers.json")
    target = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == target:
            return str(row["cik_str"]).zfill(10)
    return None


def list_10k_filings(cik: str) -> list[dict]:
    """Return [{date, accession, primary_doc}] for a CIK's 10-K filings, newest
    first, from the EDGAR submissions API.
    """
    data = _get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]
    out = []
    for form, date, accession, doc in zip(
        recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            out.append({"date": date, "accession": accession, "primary_doc": doc})
    return out


def fetch_document(cik: str, accession: str, primary_doc: str) -> str:
    """Download a filing's primary document HTML."""
    acc = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode(errors="ignore")
