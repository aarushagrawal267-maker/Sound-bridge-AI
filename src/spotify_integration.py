"""Optional Spotify integration. The core platform never depends on it.

What is used, and why it is permitted
* Track links  - https://open.spotify.com/track/{id}: public URLs, no API call.
* Spotify Embed - the official embeddable player (https://open.spotify.com/embed/track/{id}).
  It is loaded only when the user switches it on; Spotify then renders the artwork and
  any preview it chooses to offer. No playback control, no listening history.
* Web API (Client Credentials) - only if the user supplies SPOTIFY_CLIENT_ID and
  SPOTIFY_CLIENT_SECRET. Used solely for public track metadata (album artwork URLs) via
  GET /v1/tracks. Rate limits are respected (Retry-After on HTTP 429), results are cached.

What is deliberately *not* used
* Audio features / audio analysis / recommendations endpoints: Spotify restricted them for
  new applications in November 2024, and the platform must work without them anyway.
* User data or playback (would require OAuth scopes and Premium for playback).
* Any scraping. If an API call fails, the app falls back to generated artwork.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests

from .config import path

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"


def track_url(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


def embed_url(track_id: str, theme: int = 0) -> str:
    return f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme={theme}"


def credentials_available() -> bool:
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


class SpotifyClient:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None, timeout: int = 10):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        self.timeout = timeout
        self._token: str | None = None
        self._expires = 0.0
        self.cache_path: Path = path("processed_dir") / "spotify_cache.json"
        self.cache: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _auth(self) -> str:
        if self._token and time.time() < self._expires - 30:
            return self._token
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials"},
                          headers={"Authorization": f"Basic {basic}"}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self._token, self._expires = data["access_token"], time.time() + data.get("expires_in", 3600)
        return self._token

    def _get(self, url: str, params: dict | None = None, retries: int = 2) -> dict:
        for _ in range(retries + 1):
            r = requests.get(url, params=params, headers={"Authorization": f"Bearer {self._auth()}"},
                             timeout=self.timeout)
            if r.status_code == 429:
                wait = min(int(r.headers.get("Retry-After", "2")), 30)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Spotify rate limit persisted; try again later")

    def album_art(self, track_id: str) -> str | None:
        """Album artwork URL for a track (cached). Returns None on any failure."""
        if not self.enabled:
            return None
        if track_id in self.cache:
            return self.cache[track_id]
        try:
            data = self._get(f"{API}/tracks/{track_id}")
            images = data.get("album", {}).get("images", [])
            url = images[min(1, len(images) - 1)]["url"] if images else None
        except Exception:
            return None
        self.cache[track_id] = url
        try:
            self.cache_path.write_text(json.dumps(self.cache))
        except OSError:
            pass
        return url

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        try:
            data = self._get(f"{API}/search", {"q": query, "type": "track", "limit": limit})
        except Exception:
            return []
        return [
            {"id": t["id"], "name": t["name"], "artists": ", ".join(a["name"] for a in t["artists"]),
             "album": t["album"]["name"], "url": t["external_urls"].get("spotify")}
            for t in data.get("tracks", {}).get("items", [])
        ]
