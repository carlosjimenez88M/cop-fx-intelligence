"""Unit tests del cliente HTTP compartido — reintentos y degradado seguro."""

from __future__ import annotations

import httpx
import pytest

from cop_fx.data import http


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://x.com")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.unit()
def test_is_transient_only_for_5xx_and_network() -> None:
    assert http._is_transient(httpx.TimeoutException("t")) is True
    assert http._is_transient(httpx.ConnectError("c")) is True
    assert http._is_transient(_status_error(503)) is True
    assert http._is_transient(_status_error(404)) is False  # determinista, no reintentar


@pytest.mark.unit()
def test_fetch_text_returns_none_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        raise _status_error(404)

    monkeypatch.setattr(http, "_get_with_retry", _raise)
    assert http.fetch_text("https://x.com") is None


@pytest.mark.unit()
def test_fetch_text_returns_body_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", "https://x.com")
    response = httpx.Response(200, text="<html>ok</html>", request=request)
    monkeypatch.setattr(http, "_get_with_retry", lambda url, **_: response)
    assert http.fetch_text("https://x.com") == "<html>ok</html>"


@pytest.mark.unit()
def test_get_with_retry_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    request = httpx.Request("GET", "https://x.com")

    class _FakeClient:
        def __init__(self, **_: object) -> None: ...
        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_: object) -> None: ...

        def get(self, url: str) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("transient", request=request)
            return httpx.Response(200, text="recovered", request=request)

    monkeypatch.setattr(http.httpx, "Client", _FakeClient)
    resp = http._get_with_retry("https://x.com", headers={}, timeout=1.0)

    assert resp.text == "recovered"
    assert calls["n"] == 3  # 2 fallos transitorios + 1 éxito
