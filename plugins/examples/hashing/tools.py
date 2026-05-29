"""hashing plugin — cryptographic digests of text. Dependency-free."""

import hashlib as _hl

_ALGOS = ("md5", "sha1", "sha256", "sha512")


def hash_text(text: str = "", algo: str = "sha256") -> dict:
    """Compute a hash digest of a string.

    Args:
        text: input to hash.
        algo: one of md5, sha1, sha256, sha512 (default sha256).
    """
    algo = (algo or "sha256").lower()
    if algo not in _ALGOS:
        return {"success": False, "error": f"unsupported algo {algo!r}; use one of {_ALGOS}"}
    h = _hl.new(algo)
    h.update(str(text).encode("utf-8"))
    return {"success": True, "algo": algo, "hexdigest": h.hexdigest()}
