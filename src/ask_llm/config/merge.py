"""Deep-merge helper for layered configuration.

Merge order in ``ConfigLoader.load``: package default < providers.yml < user
config < env overrides. Overlay values take precedence.

``record_leaves`` supports config provenance: recording which layer supplied
each leaf value so ``config show --debug-config`` can report it.
"""

from __future__ import annotations

import copy
from typing import Any


def record_leaves(
    data: Any,
    source: str,
    provenance: dict[str, str],
    parts: tuple[str, ...] = (),
) -> None:
    """Record *source* for every leaf path in *data* (dotted notation).

    Call layers in precedence order (lowest first); later calls overwrite
    earlier labels, so the final mapping names the layer that actually won
    each key.
    """
    if isinstance(data, dict) and data:
        for k, v in data.items():
            record_leaves(v, source, provenance, (*parts, str(k)))
    elif parts:
        provenance[".".join(parts)] = source


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overlay into base. Overlay values take precedence.
    For 'providers', individual provider settings are deep-merged instead of
    replacing the entire providers dictionary.
    """
    result = copy.deepcopy(base)
    for key, overlay_val in overlay.items():
        if (
            key == "providers"
            and isinstance(overlay_val, dict)
            and isinstance(result.get(key), dict)
        ):
            if not overlay_val:
                # Explicitly empty providers dict clears everything
                result[key] = {}
            else:
                # Deep merge per-provider settings instead of replacing the whole dict
                for provider_name, provider_overlay in overlay_val.items():
                    if (
                        provider_name in result[key]
                        and isinstance(result[key][provider_name], dict)
                        and isinstance(provider_overlay, dict)
                    ):
                        result[key][provider_name] = _deep_merge(
                            result[key][provider_name], provider_overlay
                        )
                    else:
                        result[key][provider_name] = copy.deepcopy(provider_overlay)
        elif key in result and isinstance(result[key], dict) and isinstance(overlay_val, dict):
            result[key] = _deep_merge(result[key], overlay_val)
        else:
            result[key] = copy.deepcopy(overlay_val)
    return result
