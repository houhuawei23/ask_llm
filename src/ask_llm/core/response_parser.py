"""LLM response payload parsing (P4.7).

Canonical home for unwrapping JSON-wrapped payloads that some prompts/models
return despite instructions (e.g. ``{"translation": "..."}``). Moved from
``translation_exporter`` so any exporter/consumer shares one implementation.
"""

from __future__ import annotations

import contextlib
import json


def _try_parse_json_with_latex_escapes(raw: str, brace: int) -> dict | None:
    """Attempt to parse a JSON-like blob that contains invalid LaTeX escapes.

    LLMs sometimes emit ``{"translation": "text with $\\mathcal{V}$"}`` where
    ``\\mathcal`` is not a valid JSON escape.  They may also include literal
    newlines inside JSON strings.  This function:

    1. Extracts the JSON-like substring from *brace* to the last ``}``.
    2. Within string values, fixes two classes of problems:
       a. Doubles backslashes before non-JSON-escape characters
          (e.g. ``\\mathcal`` → ``\\\\mathcal``).
       b. Replaces literal control characters (newlines, tabs, etc.)
          with their JSON escape equivalents (``\\n``, ``\\t``, etc.).
    3. Parses the fixed string.
    """
    last = raw.rfind("}")
    if last <= brace:
        return None
    blob = raw[brace : last + 1]

    def _escaped(s: str, idx: int) -> bool:
        """True when s[idx] is preceded by an odd number of backslashes.

        M4/2.25: the old single-char check (``blob[i-1] != "\\\\"``) treated
        ``\\\\"`` — an escaped backslash followed by a real closing quote — as
        an escaped quote, so the string boundary was missed and every
        escaping decision downstream was wrong.
        """
        n = 0
        j = idx - 1
        while j >= 0 and s[j] == "\\":
            n += 1
            j -= 1
        return n % 2 == 1

    # Valid single-char JSON escapes after backslash
    _valid_escape_chars = set('"\\bfnrt/')
    _control_replace = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }

    fixed_parts: list[str] = []
    i = 0
    in_string = False
    while i < len(blob):
        ch = blob[i]

        # Track string boundaries (respect already-escaped quotes)
        if ch == '"' and not _escaped(blob, i):
            in_string = not in_string
            fixed_parts.append(ch)
            i += 1
        elif in_string and ch == "\\":
            # Check if this is already a valid JSON escape
            if i + 1 < len(blob) and blob[i + 1] in _valid_escape_chars:
                # Already valid — keep as-is
                fixed_parts.append(blob[i : i + 2])
                i += 2
            elif i + 1 < len(blob) and blob[i + 1] == "u":
                # Possible \uXXXX — keep as-is if valid hex
                if i + 5 < len(blob) and all(
                    c in "0123456789abcdefABCDEF" for c in blob[i + 2 : i + 6]
                ):
                    fixed_parts.append(blob[i : i + 6])
                    i += 6
                else:
                    # Invalid \u — double the backslash
                    fixed_parts.append("\\\\")
                    i += 1
            else:
                # Invalid escape (like \m in \mathcal) — double the backslash
                # so JSON sees \\m which decodes to \m
                fixed_parts.append("\\\\")
                i += 1
        elif in_string and (ch in _control_replace or ord(ch) < 0x20):
            # Literal control character inside string — replace with its JSON
            # escape (M4/2.25: any C0 control, not just \n\r\t, breaks
            # json.loads).
            fixed_parts.append(_control_replace.get(ch) or f"\\u{ord(ch):04x}")
            i += 1
        else:
            fixed_parts.append(ch)
            i += 1

    fixed = "".join(fixed_parts)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    return None


def unwrap_translation_payload(text: str) -> str:
    """
    Unwrap common JSON-wrapped translation payloads returned by some prompts/models.

    Supports payloads like:
    - {"translation": "..."}
    - ```json {"translation":"..."} ```
    - {"translation": "..."} plus trailing text (models often append notes after ``}``)
    - JSON with invalid escape sequences (e.g. ``\\mathcal`` inside string values)

    Falls back to original text when parsing fails.

    Note: Do not require ``raw.endswith("}")`` — that rejects valid JSON objects when
    the model adds characters after the closing brace, which previously left the file
    as raw JSON.

    M4: the response is only treated as a JSON payload when it *opens* with the
    object (optionally behind a fence). A translation that merely contains a
    brace — e.g. a legitimate JSON example inside the translated text — used to
    be destructively rewritten to the value of a key like ``"text"``.
    """
    original = text
    raw = text.strip()
    if not raw:
        return raw

    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff").strip()

    # Strip optional fenced code block wrapper (rstrip first: a trailing
    # newline used to make lines[-1] "" and skip the strip entirely).
    if raw.startswith("```"):
        lines = raw.rstrip().splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    # M4: require the payload to open with the object, not merely contain it.
    if not raw.startswith("{"):
        return original

    brace = 0

    obj = None

    # 1) First JSON value from first ``{`` (handles trailing garbage after ``}``).
    with contextlib.suppress(json.JSONDecodeError):
        obj, _end = json.JSONDecoder().raw_decode(raw, brace)

    # 2) Slice from first ``{`` to last ``}`` (some models emit extra junk only at end).
    if obj is None:
        with contextlib.suppress(json.JSONDecodeError):
            last = raw.rfind("}")
            if last > brace:
                obj = json.loads(raw[brace : last + 1])

    # 3) Whole blob (e.g. already trimmed to a single object).
    if obj is None and raw.startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            obj = json.loads(raw)

    # 4) JSON with invalid escape sequences (e.g. \mathcal, \beta in LaTeX).
    #    LLMs sometimes wrap translations in {"translation": "..."} even when
    #    instructed not to, and the content contains raw LaTeX whose backslashes
    #    are not valid JSON escapes.  Try to fix the escapes and re-parse.
    if obj is None:
        obj = _try_parse_json_with_latex_escapes(raw, brace)

    if isinstance(obj, dict):
        for key in ("translation", "translated_text", "content", "text", "result"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return original
