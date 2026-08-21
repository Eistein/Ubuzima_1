"""tests/test_llm_client.py — LLM fallback + structured provider-failure behaviour.

All tests inject a fake `post_fn`, so no network or API key is needed. This is the
automated 'provider failure' edge-case evidence the first-round feedback asked for.
"""
import pytest
import requests

from llm_client import LLMClient, LLMError, LLMUnavailable
from pipeline_logic import ProviderErrorKind


class FakeResp:
    def __init__(self, status=200, content="Nywa amazi menshi."):
        self.status_code = status
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def client(post_fn, models=None):
    return LLMClient(models=models or ["m-primary", "m-fallback"],
                     api_key="sk-test", post_fn=post_fn)


class TestHappyPath:
    def test_primary_success(self):
        c = client(lambda *a, **k: FakeResp(200, "Igisubizo cyiza."))
        assert c.answer("Malariya?") == "Igisubizo cyiza."


class TestFallback:
    def test_primary_5xx_then_fallback_succeeds(self):
        calls = {"n": 0}
        def post(*a, **k):
            calls["n"] += 1
            return FakeResp(503) if calls["n"] == 1 else FakeResp(200, "Byavuye kuri fallback.")
        c = client(post)
        assert c.answer("Malariya?") == "Byavuye kuri fallback."
        assert calls["n"] == 2  # primary failed, fallback used

    def test_all_models_fail_raises_unavailable(self):
        c = client(lambda *a, **k: FakeResp(503))
        with pytest.raises(LLMUnavailable):
            c.answer("Malariya?")


class TestAuthNotRetried:
    def test_auth_failure_stops_immediately(self):
        calls = {"n": 0}
        def post(*a, **k):
            calls["n"] += 1
            return FakeResp(401)
        c = client(post)
        with pytest.raises(LLMError) as ei:
            c.answer("Malariya?")
        assert ei.value.kind is ProviderErrorKind.AUTH
        assert calls["n"] == 1  # did NOT try the fallback with a bad key


class TestTimeout:
    def test_timeout_falls_back(self):
        calls = {"n": 0}
        def post(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.Timeout("slow")
            return FakeResp(200, "Nyuma yo gutinda.")
        c = client(post)
        assert c.answer("Malariya?") == "Nyuma yo gutinda."


class TestMissingKey:
    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(LLMError):
            LLMClient(api_key=None, post_fn=lambda *a, **k: FakeResp())
