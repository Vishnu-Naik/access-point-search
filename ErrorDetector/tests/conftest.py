import pytest
import requests

# This file is a global fixture for all tests

@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Disable requests.get network calls."""
    def stunted_get():
        raise RuntimeError("Network access not allowed during testing!")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())
