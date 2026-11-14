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
Simpler alternative (no Google Cloud): paste deploy/google_apps_script.gs into the Sheet,
deploy it as a web app and set

    [storage]
    webhook_url = "https://script.google.com/macros/s/.../exec"
    token = "the TOKEN value from the script"

Without either configuration every store falls back to local JSON-lines files.
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


def _webhook() -> tuple[str, str] | None:
    """Apps Script web-app receiver (deploy/google_apps_script.gs): simplest setup, no Google Cloud."""
    st = _secrets().get("storage", {})
    if st.get("webhook_url") and st.get("token"):
        return st["webhook_url"], st["token"]
    return None


def sheets_enabled() -> bool:
    if _webhook() is not None:
        return True
    try:
        return _spreadsheet() is not None
    except Exception:
        return False


class WebhookTable:
    """Same interface as SheetTable, talking to the Apps Script receiver over HTTPS."""

    def __init__(self, name: str):
        self.url, self.token = _webhook()
        self.name = name

    def _call(self, action: str, **extra) -> dict:
        import requests

        r = requests.post(self.url, json={"token": self.token, "table": self.name, "action": action, **extra},
                          timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Sheet receiver error: {data.get('error')}")
        return data

    def append(self, record: dict) -> None:
        self._call("append", record=json.loads(json.dumps(record, default=str)))

    def load(self) -> pd.DataFrame:
        rows = []
        for v in self._call("load").get("rows", []):
            try:
                rows.append(json.loads(v))
            except (TypeError, json.JSONDecodeError):
                continue
        return pd.DataFrame(rows)

    def clear(self) -> None:
        self._call("clear")


def make_table(name: str):
    return WebhookTable(name) if _webhook() is not None else SheetTable(name)


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
