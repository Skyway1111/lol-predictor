"""Tests for the dashboard-launch orchestrator runtime."""

import pytest


class TestDatabaseReadiness:
    """The migration pre-flight must not leak its connection."""

    def test_connection_closed_when_ensure_database_ready_raises(self, monkeypatch):
        """A raising ``ensure_database_ready`` still closes the pre-flight conn."""
        from league_predictor.orchestrator import runtime

        closed = {"value": False}

        class _FakeConn:
            def close(self):
                closed["value"] = True

        monkeypatch.setattr(runtime, "_check_port_available", lambda *_, **__: None)
        monkeypatch.setattr(runtime, "get_connection", lambda *_, **__: _FakeConn())

        def _boom(_conn):
            raise RuntimeError("db not ready")

        monkeypatch.setattr(runtime, "ensure_database_ready", _boom)

        with pytest.raises(RuntimeError, match="db not ready"):
            runtime.run_orchestrator(port=59999)

        assert closed["value"] is True


class TestBindHostResolution:
    """run_orchestrator must default to loopback and opt into 0.0.0.0."""

    def test_defaults_to_loopback(self, monkeypatch):
        from league_predictor.orchestrator.runtime import _resolve_bind_host

        monkeypatch.delenv("LOL_PREDICT_PUBLIC", raising=False)
        assert _resolve_bind_host(public=False) == "127.0.0.1"

    def test_flag_enables_public_bind(self, monkeypatch):
        from league_predictor.orchestrator.runtime import _resolve_bind_host

        monkeypatch.delenv("LOL_PREDICT_PUBLIC", raising=False)
        assert _resolve_bind_host(public=True) == "0.0.0.0"  # noqa: S104 - the asserted bind

    def test_env_var_enables_public_bind(self, monkeypatch):
        from league_predictor.orchestrator.runtime import _resolve_bind_host

        monkeypatch.setenv("LOL_PREDICT_PUBLIC", "true")
        assert _resolve_bind_host(public=False) == "0.0.0.0"  # noqa: S104 - the asserted bind

    def test_env_var_falsey_stays_loopback(self, monkeypatch):
        from league_predictor.orchestrator.runtime import _resolve_bind_host

        monkeypatch.setenv("LOL_PREDICT_PUBLIC", "0")
        assert _resolve_bind_host(public=False) == "127.0.0.1"

    def test_port_probe_uses_loopback_alias_for_public_bind(self, monkeypatch):
        from league_predictor.orchestrator import runtime

        captured: dict = {}

        class _FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def connect_ex(self, addr):
                captured["addr"] = addr
                return 1  # not in use

        monkeypatch.setattr(runtime.socket, "socket", lambda *_, **__: _FakeSock())
        runtime._check_port_available(9999, host="0.0.0.0")  # noqa: S104 - the bind under test
        assert captured["addr"] == ("127.0.0.1", 9999)


class TestOrchestratorPublicSurface:
    """The orchestrator's public surface is exactly ``run_orchestrator``."""

    def test_public_surface(self):
        from league_predictor import orchestrator

        assert hasattr(orchestrator, "run_orchestrator")
