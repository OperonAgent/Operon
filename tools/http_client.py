"""
Operon HTTP Client Tool.

Supports GET / POST / PUT / PATCH / DELETE with JSON or form bodies,
custom headers, Bearer token auth, and configurable timeout.
"""

import json as _json
import requests


def http_request(
    url: str,
    method: str = "GET",
    headers: dict = None,
    body: dict = None,
    params: dict = None,
    bearer_token: str = None,
    timeout: int = 20,
    content_type: str = "application/json",
) -> dict:
    """
    Make an HTTP request.

    Args:
        url           — Target URL (required)
        method        — HTTP verb: GET POST PUT PATCH DELETE (default GET)
        headers       — Extra headers dict (optional)
        body          — Request body dict — sent as JSON or form data (optional)
        params        — URL query params dict (optional)
        bearer_token  — Authorization: Bearer <token> (optional)
        timeout       — Seconds before timeout, default 20 (optional)
        content_type  — "application/json" or "application/x-www-form-urlencoded"

    Returns:
        {success, status_code, headers, body, error}
    """
    hdrs = {"User-Agent": "Operon/2.0"}
    if content_type:
        hdrs["Content-Type"] = content_type
    if bearer_token:
        hdrs["Authorization"] = f"Bearer {bearer_token}"
    if headers:
        hdrs.update(headers)

    method = method.upper()
    kwargs: dict = {
        "headers": hdrs,
        "timeout": timeout,
        "params":  params or {},
    }

    if body is not None:
        if content_type == "application/json":
            kwargs["json"] = body
        else:
            kwargs["data"] = body

    try:
        resp = requests.request(method, url, **kwargs)

        # Try to decode as JSON, fall back to text
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = resp.text

        return {
            "success":     resp.ok,
            "status_code": resp.status_code,
            "headers":     dict(resp.headers),
            "body":        resp_body,
            "error":       "" if resp.ok else f"HTTP {resp.status_code}: {resp.reason}",
        }

    except requests.exceptions.Timeout:
        return {
            "success":     False,
            "status_code": 0,
            "headers":     {},
            "body":        None,
            "error":       f"Request timed out after {timeout}s.",
        }
    except Exception as e:
        return {
            "success":     False,
            "status_code": 0,
            "headers":     {},
            "body":        None,
            "error":       f"{type(e).__name__}: {e}",
        }
