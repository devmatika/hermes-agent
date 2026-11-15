"""``gateway.profile_isolation``: outbound media never reaches into another profile's home.

Hermes profiles share one owner, so by default a turn may deliver another profile's plain files
and every profile's cache is allowlisted. A deployment whose profiles belong to different people
(one profile per employee behind one shared bot) turns isolation on; then a model-emitted
``MEDIA:``/bare path into a foreign profile home is refused in every mode, while the turn's own
files and paths outside the Hermes root keep today's behaviour.
"""
from __future__ import annotations

import pytest

from gateway.platforms.base import BasePlatformAdapter

validate = BasePlatformAdapter.validate_media_delivery_path


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """<root> with profiles alpha and beta, a file in each likely place, and the launch caches
    allowlisted the way MEDIA_DELIVERY_SAFE_ROOTS does in production."""
    root = tmp_path / "home" / ".hermes"
    files = {
        "root_cache": root / "cache" / "documents" / "root.txt",
        "root_note": root / "notes" / "root.md",
        "alpha_cache": root / "profiles" / "alpha" / "cache" / "documents" / "a.txt",
        "alpha_work": root / "profiles" / "alpha" / "workspace" / "a.md",
        "beta_cache": root / "profiles" / "beta" / "cache" / "documents" / "b.txt",
        "beta_memory": root / "profiles" / "beta" / "memories" / "MEMORY.md",
        "beta_work": root / "profiles" / "beta" / "workspace" / "b.md",
        "outside": tmp_path / "exports" / "report.pdf",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
    monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", root)
    monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", root)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root / "cache" / "documents",))
    return root, files


def _isolate(home, on: bool = True) -> None:
    (home / "config.yaml").write_text(f"gateway:\n  profile_isolation: {str(on).lower()}\n")


def _delivered(files, names):
    return {n for n in names if validate(str(files[n]))}


def _as_profile(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    class _Scope:
        def __enter__(self):
            self.token = set_hermes_home_override(home)

        def __exit__(self, *exc):
            reset_hermes_home_override(self.token)
    return _Scope()


ALL = ("root_cache", "root_note", "alpha_cache", "alpha_work", "beta_cache", "beta_memory", "beta_work", "outside")


@pytest.mark.parametrize("strict", ["0", "1"])
def test_routed_profile_delivers_only_its_own_home(tree, monkeypatch, strict):
    root, files = tree
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", strict)
    alpha = root / "profiles" / "alpha"
    _isolate(alpha)
    with _as_profile(alpha):
        delivered = _delivered(files, ALL)
    assert not delivered & {"root_cache", "root_note", "beta_cache", "beta_memory", "beta_work"}
    assert "alpha_cache" in delivered
    if strict == "0":
        assert {"alpha_work", "outside"} <= delivered


def test_root_profile_turn_cannot_reach_into_profiles(tree):
    root, files = tree
    _isolate(root)
    delivered = _delivered(files, ALL)
    assert not delivered & {"alpha_cache", "alpha_work", "beta_cache", "beta_memory", "beta_work"}
    assert {"root_cache", "root_note", "outside"} <= delivered


def test_isolation_off_keeps_shared_owner_behaviour(tree):
    root, files = tree
    alpha = root / "profiles" / "alpha"
    _isolate(alpha, on=False)
    with _as_profile(alpha):
        assert {"beta_cache", "beta_memory", "root_cache"} <= _delivered(files, ALL)
