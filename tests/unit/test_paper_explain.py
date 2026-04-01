"""Tests for paper explanation extraction and prompt resolution."""

import textwrap
from pathlib import Path

import pytest

from ask_llm.core.paper_explain import (
    _exclude_name,
    build_bundle_from_directory,
    build_bundle_from_file,
    explain_output_filename,
    match_section_key,
    resolve_prompt_path,
    split_markdown_by_headings,
    split_markdown_ordered,
)


def test_match_section_key_typo():
    assert match_section_key("Abstrct") == "abstract"
    assert match_section_key("1. Introduction") == "introduction"
    assert match_section_key("Methods") == "methods"


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
    assert "extra:" in "".join(order)
    arch_key = [k for k in order if "model" in k and k.startswith("extra:")]
    assert arch_key, order
    assert "Model text." in sections[arch_key[0]]
    assert headings[arch_key[0]] == "3 Model Architecture"


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
    assert explain_output_filename(1, "abstract") == "1-abstract.explain.md"
    assert (
        explain_output_filename(2, "extra:model-architecture")
        == "2-model-architecture.explain.md"
    )


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
