"""Persistent storage for feedback and pilot-study responses.

Free hosts such as Streamlit Community Cloud wipe local files when an app restarts, so a
deployed SoundBridge can write records to a Google Sheet the researcher owns instead.

Configuration (Streamlit secrets, e.g. the "Secrets" box on Streamlit Community Cloud or
.streamlit/secrets.toml locally):

    [storage]
    sheet_url = "https://docs.google.com/spreadsheets/d/..."

    [gcp_service_account]
    type = "service_account"
    ...                       # the JSON key of a Google service account, as TOML

The sheet must be shared (Editor) with the service account's e-mail address. Each store
writes to its own worksheet (tab) with columns: timestamp, record_json.
Without this configuration every store falls back to local JSON-lines files.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import pandas as pd


def _secrets() -> dict:
    try:
        import streamlit as st

        return {k: st.secrets[k] for k in st.secrets}
    except Exception:  # no secrets file / not running under Streamlit
        return {}


@lru_cache(maxsize=1)
def _spreadsheet():
    s = _secrets()
    if "gcp_service_account" not in s or "storage" not in s or "sheet_url" not in s["storage"]:
        return None
    import gspread

    client = gspread.service_account_from_dict(dict(s["gcp_service_account"]))
    return client.open_by_url(s["storage"]["sheet_url"])


def sheets_enabled() -> bool:
    try:
        return _spreadsheet() is not None
    except Exception:
        return False


class SheetTable:
    """Append-only table stored in one worksheet of the configured Google Sheet."""

    HEADER = ["timestamp", "record_json"]

    def __init__(self, name: str):
        sh = _spreadsheet()
        if sh is None:
            raise RuntimeError("Google Sheets storage is not configured")
        try:
            self.ws = sh.worksheet(name)
        except Exception:
            self.ws = sh.add_worksheet(title=name, rows=1000, cols=2)
            self.ws.append_row(self.HEADER)

    def append(self, record: dict) -> None:
        self.ws.append_row([time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(record, default=str)],
                           value_input_option="RAW")

    def load(self) -> pd.DataFrame:
        rows = []
        for r in self.ws.get_all_values()[1:]:
            if len(r) >= 2 and r[1]:
                try:
                    rows.append(json.loads(r[1]))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(rows)

    def clear(self) -> None:
        self.ws.clear()
        self.ws.append_row(self.HEADER)


def admin_password() -> str | None:
    """Password protecting destructive / heavy actions on a deployed app (None = local use)."""
    s = _secrets()
    return s.get("admin_password") or None
