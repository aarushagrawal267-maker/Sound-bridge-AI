"""Google Sheets backend logic, tested with an in-memory fake worksheet (no network)."""
import pandas as pd

from src import storage
from src.feedback import FeedbackEvent, FeedbackStore, StudyStore


class FakeWorksheet:
    def __init__(self):
        self.rows = []

    def append_row(self, row, value_input_option=None):
        self.rows.append(list(row))

    def get_all_values(self):
        return self.rows

    def clear(self):
        self.rows = []


class FakeSheet:
    def __init__(self):
        self.tabs = {}

    def worksheet(self, name):
        return self.tabs[name]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = FakeWorksheet()
        return self.tabs[title]


def test_stores_use_sheet_when_configured(monkeypatch):
    fake = FakeSheet()
    monkeypatch.setattr(storage, "_spreadsheet", lambda: fake)
    fb = FeedbackStore()
    assert fb.backend == "Google Sheet"
    fb.record(FeedbackEvent(session_id="s", track_idx=1, track_id="t", feedback="like"))
    StudyStore().append({"type": "pilot_response", "preferred": "hybrid"})
    assert set(fake.tabs) == {"feedback", "study_responses"}
    assert fake.tabs["feedback"].rows[0] == storage.SheetTable.HEADER
    df = FeedbackStore().load()
    assert isinstance(df, pd.DataFrame) and df.iloc[0]["feedback"] == "like"
    fb.clear()
    assert FeedbackStore().load().empty


def test_local_fallback_without_secrets(tmp_path):
    assert storage.sheets_enabled() is False
    assert FeedbackStore(tmp_path).backend.startswith("local")
    assert storage.admin_password() is None


def test_webhook_backend(monkeypatch):
    """Apps Script receiver protocol, with requests.post replaced by an in-memory fake."""
    import json

    import requests

    tabs = {}

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_post(url, json=None, timeout=None):
        assert json["token"] == "tok"
        rows = tabs.setdefault(json["table"], [])
        if json["action"] == "append":
            rows.append(__import__("json").dumps(json["record"]))
            return Resp({"ok": True})
        if json["action"] == "load":
            return Resp({"ok": True, "rows": list(rows)})
        rows.clear()
        return Resp({"ok": True})

    monkeypatch.setattr(storage, "_secrets", lambda: {"storage": {"webhook_url": "https://x", "token": "tok"}})
    monkeypatch.setattr(requests, "post", fake_post)
    fb = FeedbackStore()
    assert fb.backend == "Google Sheet"
    fb.record(FeedbackEvent(session_id="s", track_idx=3, track_id="t", feedback="dislike"))
    assert FeedbackStore().load().iloc[0]["track_idx"] == 3
    fb.clear()
    assert FeedbackStore().load().empty
    assert json.loads('{"a": 1}')["a"] == 1
