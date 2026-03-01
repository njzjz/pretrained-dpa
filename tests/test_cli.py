"""Tests for CLI model download behavior."""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.error
from pathlib import Path

import pytest

from pretrained_dpa import cli

MODEL_NAME = "DPA-3.2-5M"
MODEL_URL_1 = "https://example.com/DPA-3.2-5M-a.pt"
MODEL_URL_2 = "https://example.com/DPA-3.2-5M-b.pt"
MODEL_FILENAME = "DPA-3.2-5M.pt"


class ResponseOK:
    """Minimal context-manager response object for successful requests."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        """Read bytes like a file object."""
        return self._stream.read(size)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager without swallowing errors."""


class ResponseFail:
    """Minimal context-manager response object that fails while reading."""

    def read(self, _size: int = -1) -> bytes:
        """Raise to simulate broken connection."""
        msg = "boom"
        raise OSError(msg)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager without swallowing errors."""


def _model_map_with_hash(
    sha256: str,
    *,
    urls: list[str] | None = None,
    url: str | None = None,
) -> dict[str, dict[str, object]]:
    """Create a predictable model mapping for tests."""
    model_info: dict[str, object] = {
        "filename": MODEL_FILENAME,
        "sha256": sha256,
    }
    if urls is not None:
        model_info["urls"] = urls
    elif url is not None:
        model_info["url"] = url
    else:
        model_info["urls"] = [MODEL_URL_1]

    return {MODEL_NAME: model_info}


def test_configure_logging_sets_info_level() -> None:
    """Logging configuration should set root level to INFO."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        cli.configure_logging()
        assert root.level == logging.INFO
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_available_model_names_sorted(monkeypatch) -> None:
    """Available model names should be sorted alphabetically."""
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: {
            "DPA-3.2-5M": {},
            "DPA-3.1-3M": {},
        },
    )

    assert cli._available_model_names() == ["DPA-3.1-3M", "DPA-3.2-5M"]


def test_model_download_urls_prefers_urls_list_and_deduplicates() -> None:
    """`urls` list should be used and deduplicated in order."""
    model_info = {
        "urls": [MODEL_URL_1, MODEL_URL_1, MODEL_URL_2],
        "url": "https://example.com/legacy.pt",
    }

    assert cli._model_download_urls(model_info) == [MODEL_URL_1, MODEL_URL_2]


def test_model_download_urls_falls_back_to_url_field() -> None:
    """Single `url` should be supported for backward compatibility."""
    model_info = {"url": MODEL_URL_1}

    assert cli._model_download_urls(model_info) == [MODEL_URL_1]


def test_parser_model_choices_from_registry(monkeypatch) -> None:
    """Parser should constrain model_name choices from packaged registry."""
    monkeypatch.setattr(
        cli,
        "_available_model_names",
        lambda: ["DPA-3.1-3M", "DPA-3.2-5M"],
    )

    parser = cli.build_parser()
    args = parser.parse_args(["download", "DPA-3.1-3M"])

    assert args.model_name == "DPA-3.1-3M"


def test_parser_rejects_unknown_choice(monkeypatch) -> None:
    """Parser should reject model names outside choices list."""
    monkeypatch.setattr(cli, "_available_model_names", lambda: ["DPA-3.2-5M"])

    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["download", "NOT-EXIST"])

    assert exc.value.code == 2


def test_download_file_rejects_non_https_scheme(tmp_path) -> None:
    """Downloader should reject URLs that are not HTTPS."""
    destination = tmp_path / "cache" / MODEL_FILENAME

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        cli._download_file("http://example.com/model.pt", destination)


def test_download_file_cleans_part_on_failure(monkeypatch, tmp_path) -> None:
    """Downloader should clean up the .part file if stream copy fails."""
    destination = tmp_path / "cache" / MODEL_FILENAME
    part_path = destination.with_suffix(destination.suffix + ".part")

    def fake_urlopen(_url: str, timeout: int = 120):
        assert timeout == cli.DOWNLOAD_TIMEOUT_SECONDS
        return ResponseFail()

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OSError, match="boom"):
        cli._download_file("https://example.com/model.pt", destination)

    assert not destination.exists()
    assert not part_path.exists()


def test_probe_download_url_success(monkeypatch) -> None:
    """Probe should return latency value for reachable URL."""

    def fake_urlopen(req, timeout: int = 8):
        assert req.full_url == MODEL_URL_1
        assert timeout == cli.SOURCE_PROBE_TIMEOUT_SECONDS
        return ResponseOK(b"x")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    latency = cli._probe_download_url(MODEL_URL_1)
    assert latency is not None
    assert latency >= 0


def test_probe_download_url_failure_returns_none(monkeypatch) -> None:
    """Probe should return None for unreachable URL."""

    def fake_urlopen(_req, timeout: int = 8):
        assert timeout == cli.SOURCE_PROBE_TIMEOUT_SECONDS
        msg = "offline"
        raise urllib.error.URLError(msg)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    assert cli._probe_download_url(MODEL_URL_1) is None


def test_rank_download_urls_fastest_first(monkeypatch) -> None:
    """Ranker should place reachable URLs by latency then failed URLs."""

    def fake_probe(url: str) -> float | None:
        mapping = {
            MODEL_URL_1: 0.3,
            MODEL_URL_2: 0.1,
            "https://example.com/fail": None,
        }
        return mapping[url]

    monkeypatch.setattr(cli, "_probe_download_url", fake_probe)

    ranked = cli._rank_download_urls(
        [MODEL_URL_1, MODEL_URL_2, "https://example.com/fail"],
    )
    assert ranked == [MODEL_URL_2, MODEL_URL_1, "https://example.com/fail"]


def test_resolve_model_path_returns_existing_when_hash_matches(
    monkeypatch,
    tmp_path,
) -> None:
    """Resolver should return cached path when existing file checksum matches."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    payload = b"cached"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(payload)

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )

    resolved = cli.resolve_model_path(MODEL_NAME)

    assert resolved == model_file


def test_resolve_model_path_raises_on_unknown_model(monkeypatch) -> None:
    """Resolver should raise ValueError for unknown model aliases."""
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))

    with pytest.raises(ValueError, match="Unknown model"):
        cli.resolve_model_path("NOT-EXIST")


def test_resolve_model_path_raises_when_download_fails(monkeypatch, tmp_path) -> None:
    """Resolver should raise RuntimeError when download process returns non-zero."""
    model_dir = tmp_path / "cache"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))
    monkeypatch.setattr(cli, "download_model", lambda _name: 1)

    with pytest.raises(RuntimeError, match="Failed to resolve model"):
        cli.resolve_model_path(MODEL_NAME)


def test_download_unknown_model_uses_packaged_map(caplog) -> None:
    """Unknown model should fail and list available packaged models."""
    with caplog.at_level(logging.ERROR):
        code = cli.download_model("NOT-EXIST")

    assert code == 2
    assert "Unknown model: NOT-EXIST" in caplog.text
    assert MODEL_NAME in caplog.text


def test_download_no_source_configured(monkeypatch, caplog) -> None:
    """Downloader should fail when no source URL is configured."""
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: {
            MODEL_NAME: {
                "filename": MODEL_FILENAME,
                "sha256": "0" * 64,
            },
        },
    )

    with caplog.at_level(logging.ERROR):
        code = cli.download_model(MODEL_NAME)

    assert code == 1
    assert "No download URL configured" in caplog.text


def test_download_existing_model_skips_download(monkeypatch, tmp_path, caplog) -> None:
    """If model file exists and hash matches, downloader should not run."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    payload = b"already here"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(payload)

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(
            hashlib.sha256(payload).hexdigest(),
            urls=[MODEL_URL_1],
        ),
    )

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "already exists" in caplog.text
    assert str(model_file) in caplog.text


def test_download_tries_sources_in_ranked_order(monkeypatch, tmp_path, caplog) -> None:
    """Downloader should try ranked sources and succeed on a later source."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    payload = b"good-model"
    expected_sha = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(
            expected_sha,
            urls=[MODEL_URL_1, MODEL_URL_2],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_rank_download_urls",
        lambda _urls: [MODEL_URL_1, MODEL_URL_2],
    )

    attempted: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        attempted.append(url)
        if url == MODEL_URL_1:
            msg = "timeout"
            raise urllib.error.URLError(msg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert attempted == [MODEL_URL_1, MODEL_URL_2]
    assert model_file.read_bytes() == payload


def test_download_bad_checksum_falls_back_to_next_source(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    """Checksum failure from one source should continue to next source."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    good_payload = b"good-model"
    expected_sha = hashlib.sha256(good_payload).hexdigest()

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(
            expected_sha,
            urls=[MODEL_URL_1, MODEL_URL_2],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_rank_download_urls",
        lambda _urls: [MODEL_URL_1, MODEL_URL_2],
    )

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url == MODEL_URL_1:
            destination.write_bytes(b"corrupted")
            return
        destination.write_bytes(good_payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.WARNING):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "SHA256 verification failed" in caplog.text
    assert model_file.read_bytes() == good_payload


def test_download_all_sources_fail_returns_one(monkeypatch, tmp_path, caplog) -> None:
    """Downloader should fail when all sources are unreachable."""
    model_dir = tmp_path / "cache"

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(
            "0" * 64,
            urls=[MODEL_URL_1, MODEL_URL_2],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_rank_download_urls",
        lambda _urls: [MODEL_URL_1, MODEL_URL_2],
    )

    def fake_download(_url: str, _destination: Path) -> None:
        msg = "unreachable"
        raise urllib.error.URLError(msg)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.ERROR):
        code = cli.download_model(MODEL_NAME)

    assert code == 1
    assert "Failed to download" in caplog.text


def test_main_download_success(monkeypatch, tmp_path) -> None:
    """Main should return zero when download subcommand succeeds."""
    model_dir = tmp_path / "cache"
    payload = b"ok"

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )
    monkeypatch.setattr(cli, "_rank_download_urls", lambda _urls: _urls)

    def fake_download(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    assert cli.main(["download", MODEL_NAME]) == 0
