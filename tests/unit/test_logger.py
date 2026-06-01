"""Unit tests for the coloured logger."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from cop_fx.logger import SUCCESS, _ColouredFormatter, get_logger


@pytest.mark.unit()
def test_get_logger_returns_logger_instance() -> None:
    log = get_logger("test.basic")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.basic"


@pytest.mark.unit()
def test_get_logger_same_name_is_same_instance() -> None:
    a = get_logger("test.singleton")
    b = get_logger("test.singleton")
    assert a is b


@pytest.mark.unit()
def test_get_logger_does_not_propagate() -> None:
    log = get_logger("test.propagate")
    assert log.propagate is False


@pytest.mark.unit()
def test_success_level_value() -> None:
    assert SUCCESS == 25
    assert logging.getLevelName(SUCCESS) == "SUCCESS"


@pytest.mark.unit()
def test_success_method_emits_record() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    log = get_logger("test.success_method")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    log.success("everything is fine")  # type: ignore[attr-defined]
    output = stream.getvalue()
    assert "SUCCESS" in output
    assert "everything is fine" in output


@pytest.mark.unit()
def test_coloured_formatter_includes_level_name() -> None:
    formatter = _ColouredFormatter(use_colour=False)
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="watch out",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "WARNING" in formatted
    assert "watch out" in formatted


@pytest.mark.unit()
def test_coloured_formatter_adds_ansi_when_colour_enabled() -> None:
    formatter = _ColouredFormatter(use_colour=True)
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="boom",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    # ANSI escape sequence must be present for red (error)
    assert "\033[31m" in formatted or "\033[1;31m" in formatted


@pytest.mark.unit()
def test_coloured_formatter_no_ansi_when_colour_disabled() -> None:
    formatter = _ColouredFormatter(use_colour=False)
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="boom",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "\033[" not in formatted


@pytest.mark.unit()
@pytest.mark.parametrize(
    "level,method,msg",
    [
        (logging.DEBUG, "debug", "debug msg"),
        (logging.INFO, "info", "info msg"),
        (logging.WARNING, "warning", "warning msg"),
        (logging.ERROR, "error", "error msg"),
    ],
)
def test_standard_levels_emit_correctly(level: int, method: str, msg: str) -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.setLevel(logging.DEBUG)

    log = get_logger(f"test.level.{method}")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    getattr(log, method)(msg)
    output = stream.getvalue()
    assert msg in output
