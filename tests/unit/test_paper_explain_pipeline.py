"""Tests for paper-explain-pipeline.yml loading and key resolution."""

from pathlib import Path

import pytest

from ask_llm.config.paper_explain_pipeline import (
    FullPromptEntry,
    PaperExplainPipelineConfig,
    SectionCombo,
    load_paper_explain_pipeline,
    merged_section_labels_zh,
    parse_section_job_key,
    resolve_job_key_to_prompt_key,
    resolve_pipeline_yaml_path,
)


def test_resolve_job_key_to_prompt_key_matches_legacy_behavior():
    p = PaperExplainPipelineConfig.builtin()
    assert resolve_job_key_to_prompt_key("extra:foo", p) == "generic"
    assert resolve_job_key_to_prompt_key("appendices:h2:bar", p) == "appendices"
    assert resolve_job_key_to_prompt_key("abstract", p) == "abstract"
    assert resolve_job_key_to_prompt_key("methods", p) == "methods"
    assert resolve_job_key_to_prompt_key("full:section-full", p) == "full:section-full"


def test_builtin_has_two_full_prompts():
    p = PaperExplainPipelineConfig.builtin()
    assert len(p.resolved_full_prompts()) == 2
    assert p.filename_for_full_job_key("full:section-full") == "section-full.md"
    assert p.filename_for_full_job_key("full:outlines") == "outlines.md"
    assert p.label_zh_for_full_job_key("full:outlines")


def test_merged_section_labels_zh_override():
    # Do not mutate builtin() singleton; use model_copy with a partial labels dict.
    p = PaperExplainPipelineConfig.builtin().model_copy(
        update={"section_labels_zh": {"abstract": "自定义摘要"}}
    )
    m = merged_section_labels_zh(p)
    assert m["abstract"] == "自定义摘要"
    assert m["meta"] == "元信息"


def test_load_paper_explain_pipeline_from_repo_file():
    """Default pipeline YAML next to prompts/paper."""
    repo = Path(__file__).resolve().parents[2]
    yml = repo / "prompts" / "paper-explain-pipeline.yml"
    if not yml.is_file():
        pytest.skip("paper-explain-pipeline.yml not in repo layout")
    cfg = load_paper_explain_pipeline(str(yml))
    assert cfg.version == 1
    assert cfg.prompt_files["meta"] == "meta.md"
    assert cfg.prompt_files["generic"] == "section-generic.md"
    assert cfg.key_resolution[-1].kind == "identity"
    assert len(cfg.resolved_full_prompts()) >= 2
    assert cfg.filename_for_full_job_key("full:outlines") == "outlines.md"


def test_resolve_pipeline_yaml_path_at_prompts():
    p = resolve_pipeline_yaml_path("@prompts/paper-explain-pipeline.yml")
    assert p.is_file(), f"expected file at {p}"


def test_merge_project_yaml_with_defaults(tmp_path: Path):
    """Project file may be minimal; merged config keeps defaults for omitted keys."""
    f = tmp_path / "minimal-pipeline.yml"
    f.write_text("version: 1\n", encoding="utf-8")
    cfg = load_paper_explain_pipeline(str(f))
    assert cfg.prompt_files["meta"] == "meta.md"
    assert cfg.heading_match is not None
    assert len(cfg.heading_match) >= 5


def test_missing_pipeline_file_falls_back_to_builtin(monkeypatch, tmp_path: Path):
    missing = tmp_path / "nope.yml"
    cfg = load_paper_explain_pipeline(str(missing))
    assert cfg.prompt_files == PaperExplainPipelineConfig.builtin().prompt_files


def test_parse_section_job_key_multi_template():
    p = PaperExplainPipelineConfig.builtin()
    assert parse_section_job_key("abstract:section-abstract", p) == ("abstract", "section-abstract")
    assert parse_section_job_key("abstract", p) == (None, None)
    assert parse_section_job_key("full:outlines", p) == (None, None)
    assert parse_section_job_key("combo:x:y", p) == (None, None)


def test_resolve_template_filename_section_and_combo():
    p = PaperExplainPipelineConfig.builtin()
    p = p.model_copy(
        update={
            "section_prompts": {
                "abstract": [
                    FullPromptEntry(file="section-abstract.md"),
                    FullPromptEntry(file="alt.md"),
                ]
            },
            "section_combos": [
                SectionCombo(
                    id="ab_in",
                    keys=["abstract", "introduction"],
                    prompts=[FullPromptEntry(file="combo.md", label_zh="合并")],
                    output_stem="Abstract-Introduction",
                )
            ],
        }
    )
    assert p.resolve_template_filename_for_job("abstract:section-abstract") == "section-abstract.md"
    assert p.resolve_template_filename_for_job("abstract:alt") == "alt.md"
    assert p.resolve_template_filename_for_job("combo:ab_in:combo") == "combo.md"


def test_resolved_section_prompts_for_extra_sections():
    """Regression: extra:* keys must resolve through key_resolution to generic."""
    p = PaperExplainPipelineConfig.builtin()
    entries = p.resolved_section_prompts("extra:contents")
    assert len(entries) == 1
    assert entries[0].file == "section-generic.md"

    entries2 = p.resolved_section_prompts("appendices:h2:proofs")
    assert len(entries2) == 1
    assert entries2[0].file == "section-appendices.md"


def test_explain_output_filename_combo_slug():
    from ask_llm.core.paper_explain import explain_output_filename

    p = PaperExplainPipelineConfig.builtin()
    p = p.model_copy(
        update={
            "section_combos": [
                SectionCombo(
                    id="ab_in",
                    keys=["abstract", "introduction"],
                    prompts=[FullPromptEntry(file="combo.md")],
                    output_stem="Abstract-Introduction",
                )
            ],
        }
    )
    assert (
        explain_output_filename(3, "combo:ab_in:combo", p) == "3-abstract-introduction.explain.md"
    )
