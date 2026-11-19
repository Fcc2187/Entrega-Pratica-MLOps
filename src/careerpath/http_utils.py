"""Shared HTTP client utilities for CareerPath CLI scripts."""

from __future__ import annotations

from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    """Disallow HTTP redirects on inference endpoint calls."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_base_url(url: str) -> str:
    """Validate an ASCII HTTP/HTTPS base URL without credentials, query, or fragment."""
    if "?" in url or "#" in url or any(not 33 <= ord(char) <= 126 for char in url):
        raise ValueError("--url must use printable ASCII without spaces, query or fragment")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("--url must be a valid HTTP or HTTPS base URL") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--url must be an HTTP or HTTPS base URL")
    if port == 0 or parsed.username is not None:
        raise ValueError("--url must use a nonzero port and omit credentials")
    return url.rstrip("/")
