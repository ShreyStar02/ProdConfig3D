"""NIM configuration override helpers for CLI entrypoints."""

from typing import Optional

from ..config import AppConfig


def apply_nim_cli_overrides(
    config: AppConfig,
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
) -> None:
    config.nim.apply_overrides(
        base_url=nim_base_url,
        profile=nim_profile,
        auth_mode=nim_auth_mode,
        max_concurrency=nim_max_concurrency,
        max_retries=nim_max_retries,
        retry_backoff_seconds=nim_retry_backoff,
    )
