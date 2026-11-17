import importlib
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    old_main = sys.modules.pop("main", None)

    try:
        import main
        importlib.reload(main) if "main" in sys.modules else None
        main.init_db()

        suffix = uuid.uuid4().hex[:12]
        user_a = f"media_test_user_a_{suffix}"
        user_b = f"media_test_user_b_{suffix}"

        with main.db() as conn:
            for uid, username in [
                (user_a, f"media_user_a_{suffix}"),
                (user_b, f"media_user_b_{suffix}"),
            ]:
                conn.execute(
                    """
                    INSERT INTO users(id, username, password_salt, password_hash, created_at)
                    VALUES(?,?,?,?,?)
                    """,
                    (uid, username, "salt", "hash", main._iso(main._now())),
                )

        token_a = main._issue_session(user_a)
        token_b = main._issue_session(user_b)

        filename = f"chart_{uuid.uuid4().hex[:14]}.png"
        image_bytes = b"\x89PNG\r\n\x1a\nMEDIA_TEST"

        main.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        (main.MEDIA_DIR / filename).write_bytes(image_bytes)

        with main.db() as conn:
            conn.execute(
                "INSERT INTO media_assets(filename, user_id, created_at) VALUES(?,?,?)",
                (filename, user_a, main._iso(main._now())),
            )

        with TestClient(main.app) as client:
            yield {
                "client": client,
                "filename": filename,
                "image_bytes": image_bytes,
                "token_a": token_a,
                "token_b": token_b,
            }

    finally:
        sys.modules.pop("main", None)
        if old_main is not None:
            sys.modules["main"] = old_main


def test_media_requires_login(media_env):
    response = media_env["client"].get(
        f'/api/media/{media_env["filename"]}'
    )
    assert response.status_code == 401


def test_media_owner_can_access(media_env):
    response = media_env["client"].get(
        f'/api/media/{media_env["filename"]}',
        headers={"Authorization": f'Bearer {media_env["token_a"]}'},
    )
    assert response.status_code == 200
    assert response.content == media_env["image_bytes"]


def test_media_other_user_cannot_access(media_env):
    response = media_env["client"].get(
        f'/api/media/{media_env["filename"]}',
        headers={"Authorization": f'Bearer {media_env["token_b"]}'},
    )
    assert response.status_code == 404


def test_old_public_media_route_is_disabled(media_env):
    response = media_env["client"].get(
        f'/media/{media_env["filename"]}'
    )
    assert response.status_code == 404


def test_media_rejects_invalid_filename(media_env):
    response = media_env["client"].get(
        "/api/media/../../users.json",
        headers={"Authorization": f'Bearer {media_env["token_a"]}'},
    )
    assert response.status_code == 404
