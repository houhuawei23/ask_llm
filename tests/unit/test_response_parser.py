"""Unit tests for unwrap_translation_payload (P4.7 / M4)."""

from ask_llm.core.response_parser import unwrap_translation_payload


class TestUnwrapPayloadOpensWithObject:
    def test_plain_payload_unwrapped(self):
        assert unwrap_translation_payload('{"translation": "你好"}') == "你好"

    def test_payload_with_trailing_notes_unwrapped(self):
        response = '{"translation": "你好"}\n\n(translation complete)'
        assert unwrap_translation_payload(response) == "你好"

    def test_fenced_payload_unwrapped_with_trailing_newline(self):
        response = "```json\n{\"translation\": \"你好\"}\n```\n"
        assert unwrap_translation_payload(response) == "你好"

    def test_latex_escapes_repaired(self):
        import json

        inner = "text with $\\mathcal{V}$"
        payload = json.dumps({"translation": inner}, ensure_ascii=False)
        # Break the JSON: unescape the valid \\mathcal into raw LaTeX.
        broken = payload.replace("\\\\mathcal", "\\mathcal")
        assert unwrap_translation_payload(broken) == inner


class TestM4TranslationsContainingBracesSurvive:
    def test_translation_with_embedded_json_example_survives(self):
        """A translation whose body merely CONTAINS a JSON-looking object must
        pass through verbatim — the old first-brace extraction rewrote it to
        the value of a matching key."""
        body = (
            "下面是配置示例：\n\n"
            '{"text": "actual content", "translation": "ignored"}\n\n'
            "以上就是全部说明。"
        )
        assert unwrap_translation_payload(body) == body

    def test_translation_with_brace_but_no_json_keys_survives(self):
        body = "集合写法示例 {a, b, c} 如上所述。"
        assert unwrap_translation_payload(body) == body

    def test_prefixed_prose_is_not_stripped(self):
        """Prose before the payload no longer gets discarded with the wrapper."""
        body = '说明如下 {"translation": "你好"}'
        assert unwrap_translation_payload(body) == body


class TestEdgeCases:
    def test_empty_and_none(self):
        assert unwrap_translation_payload("") == ""
        assert unwrap_translation_payload("   ") == ""

    def test_plain_text_untouched(self):
        body = "普通的译文内容。"
        assert unwrap_translation_payload(body) == body
