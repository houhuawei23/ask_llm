"""Load and validate paper-explain-pipeline.yml (job key → prompt template mapping)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DEFAULTS_FILENAME = "paper-explain-pipeline.defaults.yml"


def _defaults_yaml_path() -> Path:
    """
    Resolve bundled ``paper-explain-pipeline.defaults.yml``.

    Tries package ``ask_llm/prompts/`` first (wheel layout), then repo-root ``prompts/``
    (editable install / ``src`` layout where prompts live beside ``src/``).
    """
    import ask_llm as _m

    pkg = Path(_m.__file__).resolve().parent
    bundled = pkg / "prompts" / _DEFAULTS_FILENAME
    if bundled.is_file():
        return bundled
    repo_prompts = pkg.parent.parent / "prompts" / _DEFAULTS_FILENAME
    if repo_prompts.is_file():
        return repo_prompts
    return bundled


def _load_raw_defaults_yaml() -> dict[str, Any]:
    path = _defaults_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled pipeline defaults missing: {path}. Reinstall ask_llm or restore {_DEFAULTS_FILENAME}."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML root in {path}: expected mapping")
    return raw


def _merge_pipeline_overrides(base: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """
    Merge project YAML on top of bundled defaults.

    - Top-level keys in ``user`` replace ``base`` (except ``section_labels_zh`` shallow merge).
    - ``None`` values in ``user`` are ignored (do not clear a default key).
    """
    out = dict(base)
    for k, v in user.items():
        if v is None:
            continue
        if k == "section_labels_zh" and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


class HeadingMatchRule(BaseModel):
    """Map ## headings to a canonical section key via fuzzy alias matching."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)


class FullPromptEntry(BaseModel):
    """One prompt template file under ``paper.prompt_dir`` plus display label."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, description="Template filename, e.g. section-abstract.md")
    label_zh: str | None = Field(
        default=None,
        description="Short Chinese label for prompts and explain/*.md preamble",
    )


class SectionCombo(BaseModel):
    """Merge multiple canonical sections into one body, then run one or more templates."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Stable id for job keys combo:<id>:<tpl_stem>",
    )
    keys: list[str] = Field(min_length=2, description="Canonical section keys in document order")
    prompts: list[FullPromptEntry] = Field(min_length=1)
    output_stem: str | None = Field(
        default=None,
        description="Optional filename stem, e.g. Abstract-Introduction → abstract-introduction.explain.md",
    )

    @field_validator("prompts")
    @classmethod
    def _unique_prompt_stems(cls, v: list[FullPromptEntry]) -> list[FullPromptEntry]:
        stems = [Path(e.file).stem for e in v]
        if len(stems) != len(set(stems)):
            raise ValueError("section combo prompts: duplicate template stems")
        return v


class KeyResolutionRule(BaseModel):
    """One rule: prefix match maps to prompt_key, or identity (use job key as prompt key)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["prefix", "identity"]
    prefix: str | None = None
    prompt_key: str | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> KeyResolutionRule:
        if self.kind == "prefix":
            if not (self.prefix and self.prompt_key):
                raise ValueError("kind=prefix requires prefix and prompt_key")
        elif self.kind == "identity":
            if self.prefix is not None or self.prompt_key is not None:
                raise ValueError("kind=identity must not set prefix or prompt_key")
        else:
            raise ValueError(f"unknown kind: {self.kind!r}")
        return self


class PaperExplainPipelineConfig(BaseModel):
    """Validated paper explain pipeline (prompt_files + key resolution + labels)."""

    model_config = ConfigDict(extra="ignore")

    version: int = Field(default=1, ge=1)
    prompt_files: dict[str, str] = Field(min_length=1)
    key_resolution: list[KeyResolutionRule]
    section_labels_zh: dict[str, str] = Field(default_factory=dict)
    heading_match: list[HeadingMatchRule] | None = Field(
        default=None,
        description="Ordered rules: first alias hit assigns heading to key (fuzzy match in paper_explain).",
    )
    section_prompts: dict[str, list[FullPromptEntry]] | None = Field(
        default=None,
        description="Optional multiple templates per canonical section key.",
    )
    section_combos: list[SectionCombo] | None = Field(
        default=None,
        description="Optional merged sections (same body concatenation) with their own templates.",
    )
    full_prompts: list[FullPromptEntry] | None = Field(
        default=None,
        description=(
            "Ordered list of full-text jobs: each uses the same paper body with a different template. "
            "If omitted, a single job uses prompt_files['full']."
        ),
    )

    @field_validator("full_prompts")
    @classmethod
    def _unique_full_template_stems(cls, v: list[FullPromptEntry] | None) -> list[FullPromptEntry] | None:
        if not v:
            return v
        stems = [Path(e.file).stem for e in v]
        if len(stems) != len(set(stems)):
            raise ValueError("full_prompts: duplicate template filename stems (same stem)")
        return v

    @field_validator("section_prompts")
    @classmethod
    def _section_prompt_stems_unique(
        cls, v: dict[str, list[FullPromptEntry]] | None
    ) -> dict[str, list[FullPromptEntry]] | None:
        if not v:
            return v
        for sk, entries in v.items():
            stems = [Path(e.file).stem for e in entries]
            if len(stems) != len(set(stems)):
                raise ValueError(f"section_prompts[{sk!r}]: duplicate template stems")
        return v

    @model_validator(mode="after")
    def _last_key_rule_is_identity(self) -> PaperExplainPipelineConfig:
        if not self.key_resolution or self.key_resolution[-1].kind != "identity":
            raise ValueError("key_resolution must end with kind: identity")
        return self

    def resolved_heading_match_rules(self) -> list[tuple[str, list[str]]]:
        """(canonical_key, aliases) in priority order."""
        if self.heading_match:
            return [(r.key, r.aliases) for r in self.heading_match]
        b = PaperExplainPipelineConfig.builtin()
        if not b.heading_match:
            return []
        return [(r.key, r.aliases) for r in b.heading_match]

    def canonical_section_keys(self) -> set[str]:
        """Keys that can appear as first segment of ``section:prompt_stem`` job ids."""
        return {k for k in self.prompt_files if k not in ("meta", "generic", "full")}

    def resolved_section_prompts(self, section_key: str) -> list[FullPromptEntry]:
        if self.section_prompts and section_key in self.section_prompts:
            return self.section_prompts[section_key]
        fn = self.prompt_files.get(section_key)
        if not fn:
            raise KeyError(f"No prompt_files entry for section: {section_key}")
        lab = self.section_labels_zh.get(section_key) or PaperExplainPipelineConfig.builtin().section_labels_zh.get(
            section_key
        )
        return [FullPromptEntry(file=fn, label_zh=lab)]

    def combo_by_id(self, combo_id: str) -> SectionCombo | None:
        if not self.section_combos:
            return None
        for c in self.section_combos:
            if c.id == combo_id:
                return c
        return None

    def combo_consumed_keys(self) -> set[str]:
        out: set[str] = set()
        if not self.section_combos:
            return out
        for c in self.section_combos:
            out.update(c.keys)
        return out

    def resolved_full_prompts(self) -> list[FullPromptEntry]:
        """Templates for full-paper jobs (explicit list, or single entry from ``prompt_files['full']``)."""
        if self.full_prompts:
            return self.full_prompts
        main = self.prompt_files.get("full", "section-full.md")
        lab = self.section_labels_zh.get("full") or PaperExplainPipelineConfig.builtin().section_labels_zh.get("full")
        return [FullPromptEntry(file=main, label_zh=lab)]

    def filename_for_full_job_key(self, job_key: str) -> str | None:
        """Resolve template filename for ``full`` or ``full:<stem>`` job keys."""
        entries = self.resolved_full_prompts()
        if not entries:
            return None
        if job_key == "full":
            return entries[0].file
        if job_key.startswith("full:"):
            stem = job_key.split(":", 1)[1]
            for e in entries:
                if Path(e.file).stem == stem:
                    return e.file
            return None
        return None

    def label_zh_for_full_job_key(self, job_key: str) -> str | None:
        """Human-readable label for full-paper jobs."""
        entries = self.resolved_full_prompts()
        if job_key == "full" and entries:
            return entries[0].label_zh or self.section_labels_zh.get("full")
        if job_key.startswith("full:"):
            stem = job_key.split(":", 1)[1]
            for e in entries:
                if Path(e.file).stem == stem:
                    return e.label_zh
        return None

    def resolve_template_filename_for_job(self, job_key: str) -> str | None:
        """Return template basename for ``prompt_dir`` for any supported job key pattern."""
        fn = self.filename_for_full_job_key(job_key)
        if fn:
            return fn
        if job_key.startswith("combo:"):
            parts = job_key.split(":")
            if len(parts) >= 3:
                combo_id, tpl_stem = parts[1], parts[2]
                c = self.combo_by_id(combo_id)
                if c:
                    for p in c.prompts:
                        if Path(p.file).stem == tpl_stem:
                            return p.file
            return None
        base, stem = parse_section_job_key(job_key, self)
        if stem is None:
            return None
        for p in self.resolved_section_prompts(base):
            if Path(p.file).stem == stem:
                return p.file
        return None

    def label_zh_for_section_job_key(self, job_key: str) -> str | None:
        """Label for ``canonical:tplstem`` section jobs."""
        base, stem = parse_section_job_key(job_key, self)
        if stem is None or base is None:
            return None
        for p in self.resolved_section_prompts(base):
            if Path(p.file).stem == stem:
                return p.label_zh
        return None

    def label_zh_for_combo_job_key(self, job_key: str) -> str | None:
        if not job_key.startswith("combo:"):
            return None
        parts = job_key.split(":")
        if len(parts) < 3:
            return None
        combo_id, tpl_stem = parts[1], parts[2]
        c = self.combo_by_id(combo_id)
        if not c:
            return None
        for p in c.prompts:
            if Path(p.file).stem == tpl_stem:
                return p.label_zh
        return None

    @classmethod
    def builtin(cls) -> PaperExplainPipelineConfig:
        """Bundled defaults from ``paper-explain-pipeline.defaults.yml`` (package ``prompts/``)."""
        global _builtin_pipeline_singleton
        if _builtin_pipeline_singleton is None:
            _builtin_pipeline_singleton = cls.model_validate(_load_raw_defaults_yaml())
        return _builtin_pipeline_singleton


_builtin_pipeline_singleton: PaperExplainPipelineConfig | None = None


def parse_section_job_key(
    job_key: str, pipeline: PaperExplainPipelineConfig | None
) -> tuple[str | None, str | None]:
    """
    Split ``canonical:prompt_stem`` into (canonical, stem) when job uses multiple templates.

    Returns (None, None) if not a section+stem pattern. Special job prefixes are excluded.
    """
    if job_key in ("meta", "full"):
        return None, None
    if job_key.startswith(("appendices:h2:", "extra:", "full:", "combo:")):
        return None, None
    if ":" not in job_key:
        return None, None
    base, stem = job_key.split(":", 1)
    if not stem:
        return None, None
    keys = (
        pipeline.canonical_section_keys()
        if pipeline
        else PaperExplainPipelineConfig.builtin().canonical_section_keys()
    )
    if base in keys:
        return base, stem
    return None, None


def slugify_output_stem(s: str) -> str:
    """User-facing filename stem → safe slug (lowercase, hyphenated)."""
    t = s.strip()
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "-", t.lower())
    t = re.sub(r"-+", "-", t).strip("-")
    return (t[:100] or "combo").strip("-")


def merged_section_labels_zh(pipeline: PaperExplainPipelineConfig) -> dict[str, str]:
    """Merge project labels on top of bundled defaults (same rule as pipeline file merge)."""
    return {**PaperExplainPipelineConfig.builtin().section_labels_zh, **pipeline.section_labels_zh}


def resolve_job_key_to_prompt_key(job_key: str, pipeline: PaperExplainPipelineConfig) -> str:
    """Map a runtime job key to a logical prompt_files key (first matching prefix rule, else identity)."""
    for rule in pipeline.key_resolution:
        if rule.kind == "prefix" and rule.prefix and job_key.startswith(rule.prefix):
            return rule.prompt_key or job_key
        if rule.kind == "identity":
            return job_key
    return job_key


def resolve_pipeline_yaml_path(pipeline_config: str, project_root: Path | None = None) -> Path:
    """
    Resolve @prompts/paper-explain-pipeline.yml or an absolute path to a concrete file.

    Search order matches ``resolve_prompt_path`` for @-paths: project root, then repo ``prompts/``,
    then package ``ask_llm/prompts/``.
    """
    base = pipeline_config.strip()
    candidates: list[Path] = []

    if base.startswith("@"):
        rel = base[1:].lstrip("/")
        root = project_root
        if root is None:
            try:
                from ask_llm.config.context import get_config

                cwd = Path.cwd()
                markers = get_config().unified_config.project_root_markers
                for marker in markers:
                    for parent in [cwd, *list(cwd.parents)]:
                        if (parent / marker).exists():
                            root = parent
                            break
                    if root:
                        break
            except Exception:
                root = None
            if not root:
                root = Path.cwd()
        candidates.append((root / rel).resolve())
    else:
        candidates.append(Path(base).expanduser().resolve())

    import ask_llm as _ask_llm

    pkg_root = Path(_ask_llm.__file__).resolve().parent
    rel_name = Path(base[1:].lstrip("/") if base.startswith("@") else base).name
    repo_prompts = (pkg_root.parent.parent / "prompts" / rel_name).resolve()
    candidates.append(repo_prompts)
    pkg_prompts = (pkg_root / "prompts" / rel_name).resolve()
    candidates.append(pkg_prompts)

    for path in candidates:
        if path.is_file():
            return path

    # Return first candidate for error messages if none exist
    return candidates[0]


def load_paper_explain_pipeline(
    pipeline_config: str,
    *,
    project_root: Path | None = None,
) -> PaperExplainPipelineConfig:
    """
    Load pipeline YAML from ``pipeline_config`` path.

    If the file is missing, returns :meth:`PaperExplainPipelineConfig.builtin` (backward compatible).
    """
    path = resolve_pipeline_yaml_path(pipeline_config, project_root=project_root)
    if not path.is_file():
        from loguru import logger

        logger.warning(f"Paper pipeline config not found: {path}; using built-in defaults")
        return PaperExplainPipelineConfig.builtin()

    raw_user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_user, dict):
        raw_user = {}
    merged = _merge_pipeline_overrides(_load_raw_defaults_yaml(), raw_user)
    return PaperExplainPipelineConfig.model_validate(merged)
