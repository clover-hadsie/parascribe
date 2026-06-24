"""Tests for logging configuration."""

from __future__ import annotations

import logging

from parascribe.config import Settings
from parascribe.log import configure_logging, debug_enabled


def _pkg() -> logging.Logger:
    return logging.getLogger("parascribe")


class TestConfigureLogging:
    def test_applies_configured_level(self):
        configure_logging(Settings(log_level="WARNING"))
        assert _pkg().level == logging.WARNING
        assert not debug_enabled()

    def test_debug_logging_forces_debug(self):
        configure_logging(Settings(log_level="WARNING", debug_logging=True))
        assert _pkg().level == logging.DEBUG
        assert debug_enabled()

    def test_owns_single_handler_and_does_not_propagate(self):
        configure_logging(Settings())
        configure_logging(Settings())  # idempotent: no handler pile-up
        assert len(_pkg().handlers) == 1
        assert _pkg().propagate is False

    def test_records_without_rid_do_not_break_formatter(self, capsys):
        configure_logging(Settings(log_level="INFO"))
        logging.getLogger("parascribe").info("no rid here")
        err = capsys.readouterr().err
        assert "rid=-" in err
        assert "no rid here" in err
