"""``x3d+<url>`` handling."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit, urlunsplit

PREFIX = "x3d+"


def is_x3d_url(obj) -> bool:
    return isinstance(obj, str) and obj.startswith(PREFIX)


def parse(url: str) -> tuple[str, dict[str, str]]:
    """Split ``x3d+https://host/bucket/ww3_4km?run=<id>&layout=ts`` into (store root, options).

    Plain paths and ``http(s)://`` / ``file://`` URLs without the prefix are accepted as well, so
    ``engine="x3d"`` works with them.
    """
    if url.startswith(PREFIX):
        url = url[len(PREFIX):]
    parts = urlsplit(url)
    opts = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    root = urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) if parts.scheme else parts.path
    return root.rstrip("/"), opts
