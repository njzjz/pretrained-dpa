"""Command line interface for pretrained_dpa."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pretrained-dpa" / "models"
DOWNLOAD_TIMEOUT_SECONDS = 120
SOURCE_PROBE_TIMEOUT_SECONDS = 8
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _load_model_map() -> dict[str, dict[str, Any]]:
    """Load model metadata from packaged JSON."""
    data_path = files("pretrained_dpa").joinpath("models.json")
    with data_path.open("r", encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
    return data


def _validate_download_url(url: str) -> None:
    """Validate that download URL uses a permitted scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        msg = f"Unsupported URL scheme for download: {parsed.scheme or '<empty>'}"
        raise ValueError(msg)


def _download_file(url: str, destination: Path) -> None:
    """Download URL content into destination atomically."""
    _validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,  # noqa: S310
            tmp_path.open("wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(destination)


def _sha256sum(path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _model_download_urls(model_info: dict[str, Any]) -> list[str]:
    """Return candidate download URLs for a model (deduplicated, ordered)."""
    candidates: list[str] = []

    raw_urls = model_info.get("urls")
    if isinstance(raw_urls, list):
        candidates.extend(item for item in raw_urls if isinstance(item, str))

    if not candidates and isinstance(model_info.get("url"), str):
        candidates.append(model_info["url"])

    seen: set[str] = set()
    unique: list[str] = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique


def _probe_download_url(url: str) -> float | None:
    """Probe one URL and return latency seconds if reachable, else None."""
    _validate_download_url(url)
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"Range": "bytes=0-0"},
        method="GET",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=SOURCE_PROBE_TIMEOUT_SECONDS):  # noqa: S310
            pass
    except (urllib.error.URLError, OSError, ValueError):
        return None

    return time.monotonic() - start


def _rank_download_urls(urls: list[str]) -> list[str]:
    """Rank candidate URLs by probe latency (fastest first)."""
    if len(urls) <= 1:
        return urls

    results: dict[str, float] = {}
    max_workers = min(4, len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_probe_download_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            latency = future.result()
            if latency is not None:
                results[url] = latency

    ranked_ok = sorted(results, key=lambda url: results[url])
    ranked_failed = [url for url in urls if url not in results]
    return ranked_ok + ranked_failed


def _available_model_names() -> list[str]:
    """Return available model names from packaged model registry."""
    return sorted(_load_model_map().keys())


def resolve_model_path(model_name: str) -> Path:
    """Resolve model alias to a verified local file, downloading if needed."""
    configure_logging()
    model_map = _load_model_map()
    model_info = model_map.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(model_map))
        msg = f"Unknown model: {model_name}. Available models: {available}"
        raise ValueError(msg)

    filename = str(model_info["filename"])
    output_path = DEFAULT_CACHE_DIR / filename

    if output_path.exists():
        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 == str(model_info["sha256"]):
            return output_path

    code = download_model(model_name)
    if code != 0:
        msg = f"Failed to resolve model '{model_name}'"
        raise RuntimeError(msg)

    return output_path


def download_model(model_name: str) -> int:
    """Download a named pretrained model if it is not already cached."""
    model_map = _load_model_map()
    model_info = model_map.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(model_map))
        LOGGER.error("Unknown model: %s", model_name)
        LOGGER.error("Available models: %s", available)
        return 2

    filename = str(model_info["filename"])
    expected_sha256 = str(model_info["sha256"])
    output_path = DEFAULT_CACHE_DIR / filename

    if output_path.exists():
        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 == expected_sha256:
            LOGGER.info("Model '%s' already exists at:", model_name)
            LOGGER.info("%s", output_path)
            return 0

        LOGGER.warning(
            "Cached file for '%s' failed SHA256 check, re-downloading...",
            model_name,
        )
        output_path.unlink(missing_ok=True)

    candidate_urls = _model_download_urls(model_info)
    if not candidate_urls:
        LOGGER.error("No download URL configured for model '%s'", model_name)
        return 1

    ranked_urls = _rank_download_urls(candidate_urls)
    if len(ranked_urls) > 1:
        LOGGER.info(
            "Selecting fastest source among %d candidates...",
            len(ranked_urls),
        )

    LOGGER.info("Downloading '%s'...", model_name)
    for idx, download_url in enumerate(ranked_urls, start=1):
        LOGGER.info("Attempt %d/%d: %s", idx, len(ranked_urls), download_url)
        try:
            _download_file(download_url, output_path)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            LOGGER.warning("Download attempt failed from %s: %s", download_url, exc)
            continue

        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 != expected_sha256:
            output_path.unlink(missing_ok=True)
            LOGGER.warning(
                "Downloaded '%s' from %s but SHA256 verification failed.",
                model_name,
                download_url,
            )
            LOGGER.warning("Expected: %s", expected_sha256)
            LOGGER.warning("Actual:   %s", actual_sha256)
            continue

        LOGGER.info("Downloaded '%s' to:", model_name)
        LOGGER.info("%s", output_path)
        return 0

    LOGGER.error("Failed to download '%s' from all configured sources.", model_name)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pretrained-dpa",
        description="Utilities for pretrained DPA model files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download a pretrained model",
    )
    download_parser.add_argument(
        "model_name",
        choices=_available_model_names(),
        help="Model name, e.g. DPA-3.2-5M",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return exit code."""
    configure_logging()

    parser = build_parser()
    args = parser.parse_args(argv)

    return download_model(args.model_name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())  # pragma: no cover
