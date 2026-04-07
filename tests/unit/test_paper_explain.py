"""Tests for paper explanation extraction and prompt resolution."""

import textwrap
from pathlib import Path

import pytest

from ask_llm.config.paper_explain_pipeline import PaperExplainPipelineConfig
from ask_llm.core.paper_explain import (
    _exclude_name,
    build_bundle_from_directory,
    build_bundle_from_file,
    expand_appendices_into_h2_jobs,
    explain_output_filename,
    match_section_key,
    normalize_paper_explain_response,
    resolve_prompt_key,
    resolve_prompt_path,
    split_markdown_by_headings,
    split_markdown_ordered,
)


def test_match_section_key_typo():
    assert match_section_key("Abstrct") == "abstract"
    assert match_section_key("1. Introduction") == "introduction"
    assert match_section_key("Methods") == "methods"
    assert match_section_key("Related Work") == "related_work"
    assert match_section_key("Model Architecture") == "model_architecture"


def test_split_markdown_by_headings():
    md = textwrap.dedent(
        """\
        # Title Here

        preamble ignored

        ## Abstract

        Abs body.

        ## Introduction

        Intro body.

        ## Methods

        Methods body.
        """
    )
    sections, _unmatched = split_markdown_by_headings(md)
    assert "abstract" in sections
    assert "Abs body." in sections["abstract"]
    assert "introduction" in sections
    assert "methods" in sections
    assert sections["methods"] == "Methods body."


def test_split_markdown_ordered_extra_sections():
    md = textwrap.dedent(
        """\
        # Title Here

        ## Abstract

        Abs.

        ## 3 Model Architecture

        Model text.

        ## 4 Why Self-Attention

        Why text.
        """
    )
    sections, order, headings = split_markdown_ordered(md)
    assert sections["abstract"] == "Abs."
    assert "model_architecture" in order
    assert "Model text." in sections["model_architecture"]
    assert headings["model_architecture"] == "3 Model Architecture"


def test_h2_split_keeps_h3_in_same_section():
    """Only ``##`` splits sections; ``###``/deeper stay in the same explain block."""
    md = textwrap.dedent(
        """\
        ## 3 Model Architecture

        intro.

        ### 3.1 Encoder and Decoder

        enc body.

        #### 3.2.1 Scaled Dot-Product

        dot body.

        ## 4 Why Self-Attention

        why body.
        """
    )
    sections, order, headings = split_markdown_ordered(md)
    assert len(order) == 2
    k_arch = order[0]
    k_why = order[1]
    assert headings[k_arch] == "3 Model Architecture"
    b = sections[k_arch]
    assert "### 3.1 Encoder" in b
    assert "#### 3.2.1 Scaled Dot-Product" in b
    assert "enc body." in b and "dot body." in b
    assert "why body." not in b
    assert "why body." in sections[k_why]


def test_explain_output_filename():
    assert explain_output_filename(0, "meta") == "0-meta.explain.md"
    assert explain_output_filename(9, "full") == "9-full.explain.md"
    assert explain_output_filename(3, "full:outlines") == "3-full-outlines.explain.md"
    assert explain_output_filename(1, "abstract") == "1-abstract.explain.md"
    assert (
        explain_output_filename(2, "extra:model-architecture")
        == "2-model-architecture.explain.md"
    )
    assert (
        explain_output_filename(99, "appendices:h2:proof-of-theorem-1")
        == "d-appendices-proof-of-theorem-1.explain.md"
    )


def test_resolve_prompt_key_appendix_h2():
    assert resolve_prompt_key("appendices:h2:foo-bar") == "appendices"


def test_resolve_prompt_key_with_pipeline_config():
    pl = PaperExplainPipelineConfig.builtin()
    assert resolve_prompt_key("appendices:h2:foo-bar", pl) == "appendices"
    assert resolve_prompt_key("extra:x", pl) == "generic"


def test_expand_appendices_into_h2_jobs_no_h2():
    md = "Intro line.\n\nNo heading here."
    jobs = expand_appendices_into_h2_jobs(md)
    assert len(jobs) == 1
    assert jobs[0][0] == "appendices"
    assert "Intro line" in jobs[0][1]
    assert jobs[0][2] == ""


def test_normalize_paper_explain_response_json_to_markdown():
    raw = '{"a": "hello", "b": {"c": "nested"}}'
    out = normalize_paper_explain_response(raw)
    assert "## a" in out
    assert "hello" in out
    assert "### c" in out or "## c" in out
    assert "nested" in out
    assert out == normalize_paper_explain_response("```json\n" + raw + "\n```")


def test_normalize_paper_explain_response_plain_markdown_unchanged():
    md = "## 小结\n\n正文 **粗体**。\n"
    assert normalize_paper_explain_response(md) == md


def test_expand_appendices_into_h2_jobs_multiple_h2():
    md = textwrap.dedent(
        """\
        Preamble before first H2.

        ## Part A

        Body A.

        ## Part B

        Body B.
        """
    )
    jobs = expand_appendices_into_h2_jobs(md)
    assert len(jobs) == 2
    assert jobs[0][0].startswith("appendices:h2:")
    assert jobs[0][2] == "Part A"
    assert "Preamble" in jobs[0][1] and "Body A." in jobs[0][1]
    assert "Body B." in jobs[1][1]


def test_exclude_name():
    assert _exclude_name("x_trans.md") is True
    assert _exclude_name("paper-References.md") is True
    assert _exclude_name("paper-Appendix.md") is True
    assert _exclude_name("paper.md") is False


def test_resolve_prompt_bundled():
    p = resolve_prompt_path("@prompts/paper", "meta.md")
    assert p.is_file()
    assert p.name == "meta.md"


def test_build_bundle_from_minimal_file(tmp_path: Path):
    f = tmp_path / "p.md"
    f.write_text(
        textwrap.dedent(
            """\
            # My Paper Title

            ## Abstract

            Short.

            ## Methods

            Do things.
            """
        ),
        encoding="utf-8",
    )
    b = build_bundle_from_file(f)
    assert b.paper_title == "My Paper Title"
    assert "abstract" in b.sections
    assert "methods" in b.sections


@pytest.mark.skipif(
    not Path(
        "/home/hhw/Desktop/00_Personal/my_scripts/output2/20170612-Arxiv-Attention-Is-All-You-Need"
    ).is_dir(),
    reason="Sample arxiv2md directory not present",
)
def test_build_bundle_from_real_arxiv_dir():
    d = Path("/home/hhw/Desktop/00_Personal/my_scripts/output2/20170612-Arxiv-Attention-Is-All-You-Need")
    b = build_bundle_from_directory(d)
    assert "Attention" in b.paper_title
    assert len(b.full_text) > 100
    assert b.main_path is not None
