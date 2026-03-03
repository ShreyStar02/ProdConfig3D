"""NIM endpoint probe utilities."""

from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx


async def probe_nim_endpoint(base_url: str, headers: Dict[str, str], timeout_seconds: float) -> Dict[str, Any]:
    candidates = ["", "/health", "/models", "/v1/health", "/v1/models"]
    timeout = max(0.5, float(timeout_seconds))
    base = str(base_url).rstrip("/") + "/"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        last_error: Optional[str] = None
        for path in candidates:
            url = urljoin(base, path.lstrip("/"))
            try:
                response = await client.get(url, headers=headers)
                return {
                    "reachable": True,
                    "status": response.status_code,
                    "path": path or "/",
                }
            except Exception as exc:
                last_error = str(exc)

    return {
        "reachable": False,
        "status": None,
        "path": "-",
        "error": last_error or "no response",
    }
