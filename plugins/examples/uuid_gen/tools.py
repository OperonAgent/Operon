"""uuid_gen plugin — UUID and random-token generation. Dependency-free."""

import uuid as _uuid
import secrets as _secrets


def uuid_generate(count: int = 1) -> dict:
    """Generate one or more UUID v4 strings.

    Args:
        count: how many UUIDs to generate (1-100).
    """
    count = max(1, min(int(count), 100))
    ids = [str(_uuid.uuid4()) for _ in range(count)]
    return {"success": True, "count": count, "uuids": ids}


def random_token(nbytes: int = 16, fmt: str = "hex") -> dict:
    """Generate a cryptographically-secure random token.

    Args:
        nbytes: number of random bytes (1-128).
        fmt:    "hex" | "urlsafe".
    """
    nbytes = max(1, min(int(nbytes), 128))
    token = (_secrets.token_urlsafe(nbytes) if fmt == "urlsafe"
             else _secrets.token_hex(nbytes))
    return {"success": True, "format": fmt, "bytes": nbytes, "token": token}
