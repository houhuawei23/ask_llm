"""Pre-run API health probes (pure logic, no console/typer I/O).

One probe per (provider, model): a tiny real API call via the provider
adapter's ``test_connection()`` (~10 tokens), with a shortened timeout so a
dead endpoint bounds the probe instead of stalling for the run timeout.
Extracted from ``batch_service._validate_models`` so batch, and the new
``cli_session.preflight_or_exit`` gate used by trans/paper/format, share a
single implementation.

The interactive continue/exit prompt lives in the CLI layer
(``config.cli_session``); this module only classifies and probes.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from ask_llm.utils.api_key_gate import (
    PROVIDERS_WITHOUT_API_KEYS,
    api_key_is_missing_or_unresolved,
    provider_env_var_name,
)
from ask_llm.utils.provider_cache import ProviderAdapterCache

if TYPE_CHECKING:
    from ask_llm.config.manager import ConfigManager
    from ask_llm.core.batch_models import ModelConfig

# Probe timeout: bounds a dead endpoint to ~10s per attempt (the OpenAI SDK's
# internal retries can multiply the wall time, hence the small value).
PREFLIGHT_TIMEOUT_SECONDS: float = 10.0

# Default bound for user-facing failure reasons: one provider's HTML error
# dump must not flood the console/table/retry-log lines.
MAX_REASON_CHARS = 120


@dataclass(frozen=True)
class PreflightOutcome:
    """Result of probing one (provider, model) pair before a run."""

    provider: str
    model: str
    ok: bool
    latency: float | None  # seconds; None when the probe failed
    message: str  # "1.23s" on success; failure reason otherwise


def shorten_message(message: str, max_chars: int = MAX_REASON_CHARS) -> str:
    """Collapse whitespace and truncate *message* for user-facing display.

    Shared one-line reason formatter for the preflight table, the runner's
    retry WARNING lines, and the executor's per-attempt progress labels, so
    the truncation semantics cannot drift between them.
    """
    flat = " ".join(str(message).split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


def check_provider_pair(
    config_manager: ConfigManager,
    provider: str,
    model: str,
    *,
    timeout_seconds: float = PREFLIGHT_TIMEOUT_SECONDS,
    sampling_overrides: ModelConfig | None = None,
) -> PreflightOutcome:
    """Probe one (provider, model) pair with a tiny real API call.

    Args:
        config_manager: Active config manager (per-provider overrides are
            honored via ``get_provider_config``; the manager itself is never
            mutated — audit 2.5).
        provider: Provider name.
        model: Model name to probe.
        timeout_seconds: Short timeout installed on the probe config so a
            dead endpoint cannot stall for the run timeout.
        sampling_overrides: Optional batch-style ``ModelConfig`` whose
            temperature/top_p/max_tokens are applied to the probe so the
            probe matches the run request shape.

    Returns:
        A :class:`PreflightOutcome`; never raises.
    """
    if provider not in config_manager.unified_config.providers:
        return PreflightOutcome(provider, model, False, None, "Provider not found")

    pc = config_manager.get_provider_config(provider)

    # Keyless providers (e.g. ollama) are exempt from the key check but still
    # probed — a local probe is free and genuinely useful.
    if provider not in PROVIDERS_WITHOUT_API_KEYS and api_key_is_missing_or_unresolved(pc.api_key):
        return PreflightOutcome(
            provider,
            model,
            False,
            None,
            f"API key not configured (set {provider_env_var_name(provider)})",
        )

    if pc.models and model not in pc.models:
        return PreflightOutcome(
            provider,
            model,
            False,
            None,
            f"Model not available. Available: {', '.join(pc.models)}",
        )

    default_model = model or (pc.models[0] if pc.models else "")
    if not default_model:
        return PreflightOutcome(
            provider, model, False, None, "No model available for this provider"
        )

    overrides: dict[str, Any] = {"timeout": float(timeout_seconds)}
    if sampling_overrides is not None:
        if sampling_overrides.temperature is not None:
            overrides["api_temperature"] = sampling_overrides.temperature
        if sampling_overrides.top_p is not None:
            overrides["api_top_p"] = sampling_overrides.top_p
        if sampling_overrides.max_tokens is not None:
            overrides["max_tokens"] = sampling_overrides.max_tokens
    probe_config = pc.model_copy(update=overrides)

    try:
        adapter = ProviderAdapterCache.get(probe_config, default_model=default_model)
    except Exception as e:
        return PreflightOutcome(
            provider, model, False, None, shorten_message(f"Failed to create provider adapter: {e}")
        )

    try:
        success, message, latency = adapter.test_connection()
    except Exception as e:  # defensive: test_connection is documented not to raise
        return PreflightOutcome(
            provider, model, False, None, shorten_message(f"Connection test crashed: {e}")
        )

    if not success:
        return PreflightOutcome(
            provider, model, False, None, shorten_message(f"Connection test failed: {message}")
        )
    return PreflightOutcome(provider, model, True, latency, f"{latency:.2f}s")


def check_provider_pairs(
    config_manager: ConfigManager,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout_seconds: float = PREFLIGHT_TIMEOUT_SECONDS,
    max_workers: int = 8,
) -> list[PreflightOutcome]:
    """Probe many (provider, model) pairs concurrently, preserving input order.

    Duplicate pairs are probed once (first occurrence wins); results are
    returned in the order the unique pairs first appear in *pairs*.
    """
    unique = list(dict.fromkeys(pairs))
    if not unique:
        return []

    workers = max(1, min(max_workers, len(unique)))
    ordered: list[PreflightOutcome | None] = [None] * len(unique)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ask-llm-preflight") as pool:
        futures = {
            pool.submit(
                check_provider_pair,
                config_manager,
                provider,
                model,
                timeout_seconds=timeout_seconds,
            ): i
            for i, (provider, model) in enumerate(unique)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                ordered[index] = future.result()
            except Exception as e:  # defensive: check_provider_pair never raises
                provider, model = unique[index]
                logger.warning(f"Preflight probe crashed for {provider}/{model}: {e}")
                ordered[index] = PreflightOutcome(
                    provider, model, False, None, shorten_message(f"Probe crashed: {e}")
                )
    return [outcome for outcome in ordered if outcome is not None]
