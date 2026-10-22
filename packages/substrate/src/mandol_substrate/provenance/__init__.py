"""Explicit source provenance for reviewed mechanical package relocation."""
import hashlib


def qualified_source_hash(data: bytes, *, original: str, promoted: str) -> str:
    """Alias only one exact reviewed artifact; every subsequent edit invalidates it.

    This does not claim arbitrary refactors are semantically equivalent. The
    adapter supplies its independently reviewed pair of source artifact hashes.
    """
    observed = hashlib.sha256(data).hexdigest()
    return original if observed == promoted else observed
