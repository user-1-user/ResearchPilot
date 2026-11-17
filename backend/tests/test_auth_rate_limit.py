import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture
def auth_module(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    old_main = sys.modules.pop("main", None)
    try:
        import main
        main._reset_login_rate_limits()
        yield main
    finally:
        sys.modules.pop("main", None)
        if old_main is not None:
            sys.modules["main"] = old_main


def test_ip_allows_ten_requests_and_blocks_eleventh(auth_module, monkeypatch):
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: 100.0)

    for i in range(10):
        auth_module._check_login_rate_limit("192.0.2.10", f"user{i}")

    with pytest.raises(HTTPException) as exc:
        auth_module._check_login_rate_limit("192.0.2.10", "another-user")

    assert exc.value.status_code == 429


def test_ip_limit_expires_after_one_minute(auth_module, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: now[0])

    for i in range(10):
        auth_module._check_login_rate_limit("192.0.2.11", f"user{i}")

    now[0] = 160.0
    auth_module._check_login_rate_limit("192.0.2.11", "new-user")


def test_five_failures_lock_ip_username_pair(auth_module, monkeypatch):
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: 100.0)

    for _ in range(5):
        auth_module._check_login_rate_limit("192.0.2.12", "Alice")
        auth_module._record_login_failure("192.0.2.12", "Alice")

    with pytest.raises(HTTPException) as exc:
        auth_module._check_login_rate_limit("192.0.2.12", "Alice")

    assert exc.value.status_code == 429


def test_lock_is_scoped_to_ip_and_username(auth_module, monkeypatch):
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: 100.0)

    for _ in range(5):
        auth_module._check_login_rate_limit("192.0.2.13", "Alice")
        auth_module._record_login_failure("192.0.2.13", "Alice")

    auth_module._check_login_rate_limit("192.0.2.14", "Alice")
    auth_module._check_login_rate_limit("192.0.2.13", "Bob")


def test_success_clears_failure_lock(auth_module, monkeypatch):
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: 100.0)

    for _ in range(5):
        auth_module._check_login_rate_limit("192.0.2.15", "Alice")
        auth_module._record_login_failure("192.0.2.15", "Alice")

    auth_module._clear_login_failures("192.0.2.15", "Alice")
    auth_module._check_login_rate_limit("192.0.2.15", "Alice")


def test_failure_lock_expires_after_fifteen_minutes(auth_module, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: now[0])

    for _ in range(5):
        auth_module._check_login_rate_limit("192.0.2.16", "Alice")
        auth_module._record_login_failure("192.0.2.16", "Alice")

    now[0] = 1000.0
    auth_module._check_login_rate_limit("192.0.2.16", "Alice")


def test_login_endpoint_success(auth_module):
    from fastapi.testclient import TestClient

    with TestClient(auth_module.app) as client:
        username = "integration_user"
        password = "StrongPass123"

        register = client.post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        assert register.status_code == 200

        login = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        assert login.status_code == 200
        assert login.json()["token"]
        assert login.json()["username"] == username


def test_login_endpoint_wrong_password_and_lockout(auth_module):
    from fastapi.testclient import TestClient

    with TestClient(auth_module.app) as client:
        username = "lockout_integration_user"
        password = "StrongPass123"

        register = client.post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        assert register.status_code == 200

        for _ in range(5):
            response = client.post(
                "/api/auth/login",
                json={"username": username, "password": "WrongPassword"},
            )
            assert response.status_code == 401
            assert response.json()["detail"] == "\u7528\u6237\u540d\u6216\u5bc6\u7801\u9519\u8bef"

        blocked = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        assert blocked.status_code == 429
        assert blocked.headers.get("Retry-After")


def test_login_endpoint_ip_rate_limit(auth_module):
    from fastapi.testclient import TestClient

    with TestClient(auth_module.app) as client:
        for i in range(10):
            response = client.post(
                "/api/auth/login",
                json={
                    "username": f"unknown_user_{i}",
                    "password": "WrongPassword",
                },
            )
            assert response.status_code == 401

        blocked = client.post(
            "/api/auth/login",
            json={"username": "eleventh_user", "password": "WrongPassword"},
        )
        assert blocked.status_code == 429

def test_ip_tracking_has_hard_memory_limit(auth_module, monkeypatch):
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: 100.0)

    limit = auth_module._LOGIN_IP_TRACKED_MAX

    for i in range(limit):
        auth_module._check_login_rate_limit(f"198.51.{i // 256}.{i % 256}", "user")

    assert len(auth_module._LOGIN_REQUESTS) == limit

    with pytest.raises(HTTPException) as exc:
        auth_module._check_login_rate_limit("203.0.113.250", "new-user")

    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "60"
    assert len(auth_module._LOGIN_REQUESTS) == limit
    assert "203.0.113.250" not in auth_module._LOGIN_REQUESTS

