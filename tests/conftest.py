"""Unit fixtures must never connect to an operator's default deployment."""
import pytest


@pytest.fixture(autouse=True)
def isolate_default_services(monkeypatch):
    # Real transport tests opt in through MANDOL_TEST_QDRANT or explicit URLs.
    for name in ('QDRANT_URL', 'EMBEDDING_BASE_URL', 'SPLADE_URL', 'RERANKER_BASE_URL'):
        monkeypatch.setenv('CHINASOFT_' + name, 'http://127.0.0.1:1')
