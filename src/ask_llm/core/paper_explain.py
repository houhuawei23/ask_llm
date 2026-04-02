"""Paper explanation: extract sections from Markdown / arxiv2md-beta dirs, resolve prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

# Standard section keys (order for section runs)
SECTION_ORDER: list[str] = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendices",
]

# Heading text (normalized) -> canonical key; values are substrings or regex-friendly tokens
# More specific keys first (insertion order) to avoid e.g. "related work" matching introduction.
_HEADING_ALIASES: dict[str, list[str]] = {
    "abstract": ["abstract", "abstrct", "summary"],
    "related_work": [
        "related work",
        "related works",
        "prior work",
        "literature review",
        "background and related work",
        "literature",
    ],
    "introduction": [
        "introduction",
        "introducution",
        "intro",
        "background",
    ],
    "model_architecture": [
        "model architecture",
        "network architecture",
        "architecture",
        "overview of the model",
        "model design",
        "proposed architecture",
        "proposed model",
        "system architecture",
        "the proposed method",
    ],
    "methods": [
        "methods",
        "method",
        "methodology",
        "approach",
        "experimental setup",
    ],
    "results": [
        "results",
        "experiments",
        "evaluation",
        "empirical",
    ],
    "discussion": ["discussion", "analysis", "limitations"],
    "conclusion": [
        "conclusion",
        "conclusions",
        "concluding",
        "concluding remarks",
    ],
    "references": [
        "references",
        "reference",
        "bibliography",
        "literature cited",
    ],
    "appendices": [
        "appendix",
        "appendices",
        "supplementary",
        "supplemental",
        "appendix a",
    ],
}

_PROMPT_FILES: dict[str, str] = {
    "meta": "meta.md",
    "abstract": "section-abstract.md",
    "introduction": "section-introduction.md",
    "related_work": "section-related-work.md",
    "model_architecture": "section-model-architecture.md",
    "methods": "section-methods.md",
    "results": "section-results.md",
    "discussion": "section-discussion.md",
    "conclusion": "section-conclusion.md",
    "references": "section-references.md",
    "appendices": "section-appendices.md",
    "full": "section-full.md",
    "generic": "section-generic.md",
}

SECTION_LABELS_ZH: dict[str, str] = {
    "meta": "元信息",
    "abstract": "Abstract (摘要)",
    "introduction": "Introduction (引言)",
    "related_work": "Related Work (相关工作)",
    "model_architecture": "Model Architecture (模型架构)",
    "methods": "Methods (方法)",
    "results": "Results (结果)",
    "discussion": "Discussion (讨论)",
    "conclusion": "Conclusion (结论)",
    "references": "References (参考文献)",
    "appendices": "Appendices (附录)",
    "full": "全文总体分析",
}


def _normalize_heading_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^[\d.]+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
    return s.strip()


def match_section_key(heading_text: str) -> str | None:
    """Map a markdown heading line (without #) to a canonical section key."""
    norm = _normalize_heading_text(heading_text)
    if not norm:
        return None
    for key, aliases in _HEADING_ALIASES.items():
        for a in aliases:
            if norm == a or norm.startswith(a + " ") or norm.endswith(" " + a):
                return key
            if a in norm and len(norm) <= len(a) + 12:
                return key
    return None


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Level-2 headings only (`## title`, not `###`): major paper sections for explain jobs
_H2_SECTION_LINE_RE = re.compile(r"^##\s+(.+?)\s*$")


def _slug_for_extra_heading(title: str) -> str:
    s = re.sub(r"^[\d.]+\s*", "", title.strip())
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", s.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:60] or "section").strip("-")


def _extra_key_from_heading(title: str, used: set[str]) -> str:
    base = _slug_for_extra_heading(title)
    k = f"extra:{base}"
    if k not in used:
        used.add(k)
        return k
    n = 2
    while True:
        k2 = f"extra:{base}-{n}"
        if k2 not in used:
            used.add(k2)
            return k2
        n += 1


def _parse_markdown_heading_blocks(text: str) -> list[tuple[str, str]]:
    """
    Split text into (heading_text, body) using **level-2 headings only** (``##``).

    Deeper headings (``###``, ``####``, …) stay inside the same section body so one
    explain job covers a whole logical section (e.g. all of "3 Model Architecture"
    including 3.1–3.5), not each subsection separately.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    cur_title: str | None = None
    cur_lines: list[str] = []
    for line in lines:
        m = _H2_SECTION_LINE_RE.match(line)
        if m:
            if cur_title is not None:
                blocks.append((cur_title, cur_lines))
            cur_title = m.group(1).strip()
            cur_lines = []
        else:
            if cur_title is not None:
                cur_lines.append(line)
    if cur_title is not None:
        blocks.append((cur_title, cur_lines))
    out: list[tuple[str, str]] = []
    for h, ls in blocks:
        body = "\n".join(ls).strip()
        if body:
            out.append((h, body))
    return out


def split_markdown_ordered(text: str) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """
    Split markdown into sections by **``##`` (level-2) headings only**, preserving order.

    Subsections (``###`` …) are **not** split out; they remain in the parent section body.

    Unmapped headings become keys ``extra:<slug>`` so each block is kept.

    Returns:
        sections: key -> body
        section_order: keys in first-occurrence order
        section_headings: key -> original heading line (first occurrence)
    """
    sections: dict[str, str] = {}
    section_order: list[str] = []
    section_headings: dict[str, str] = {}
    used_extra: set[str] = set()

    for heading, body in _parse_markdown_heading_blocks(text):
        mk = match_section_key(heading)
        if mk:
            key = mk
        else:
            key = _extra_key_from_heading(heading, used_extra)
        if key not in section_headings:
            section_headings[key] = heading
        if key in sections:
            sections[key] = sections[key] + "\n\n" + body
        else:
            sections[key] = body
            section_order.append(key)

    return sections, section_order, section_headings


def split_markdown_by_headings(text: str) -> tuple[dict[str, str], list[str]]:
    """
    Backward-compatible wrapper: same sections dict, ``unmatched`` is always empty
    (non-standard headings are stored under ``extra:*`` keys).
    """
    sections, _, _ = split_markdown_ordered(text)
    return sections, []


def _first_heading_or_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = _HEADING_LINE_RE.match(line)
        if m:
            return m.group(2).strip()
    return fallback


def _exclude_name(name: str) -> bool:
    """Exclude translation and sidecar markdown from main-body selection."""
    lower = name.lower()
    if not lower.endswith(".md"):
        return True
    if "_trans.md" in lower or lower.endswith("_trans.md"):
        return True
    if "-references.md" in lower or lower.endswith("-references.md"):
        return True
    return bool("-appendix.md" in lower or "-appendices.md" in lower)


def _pick_main_md(directory: Path) -> Path:
    """Choose main paper markdown under arxiv2md-beta output directory."""
    all_md = sorted(directory.glob("*.md"))
    candidates = [p for p in all_md if not _exclude_name(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No suitable main .md in {directory}")

    dir_name = directory.name.lower()
    for p in candidates:
        if p.stem.lower() == dir_name:
            logger.info(f"Selected main markdown (name match): {p}")
            return p
    # Prefer longest stem (often full title slug)
    candidates.sort(key=lambda x: len(x.stem), reverse=True)
    chosen = candidates[0]
    logger.info(f"Selected main markdown (heuristic): {chosen}")
    return chosen


def _find_sidecar(directory: Path, glob_pat: str) -> Path | None:
    """Find first matching sidecar file (e.g. *-References.md)."""
    found = list(directory.glob(glob_pat))
    if not found:
        return None
    found.sort(key=lambda p: len(p.name))
    return found[0]


@dataclass
class PaperBundle:
    """Resolved paper content for explanation.

    ``section_order`` lists keys in document order (standard keys and ``extra:*``).
    ``section_headings`` maps each key to the first-seen original heading text.
    """

    paper_title: str
    meta_text: str
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    source_description: str = ""
    main_path: Path | None = None
    section_order: list[str] = field(default_factory=list)
    section_headings: dict[str, str] = field(default_factory=dict)


def load_paper_yml_meta(yml_path: Path) -> tuple[str, str]:
    """Return (title, meta_markdown_block)."""
    raw = yml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    paper = data.get("paper") or data
    title = (paper.get("title") or "").strip() or "Unknown title"
    # Compact meta for LLM
    lines = [f"title: {title}"]
    authors = paper.get("authors")
    if isinstance(authors, list):
        names = []
        for a in authors:
            if isinstance(a, dict) and a.get("name"):
                names.append(str(a["name"]))
            elif isinstance(a, str):
                names.append(a)
        if names:
            lines.append("authors: " + ", ".join(names))
    pub = paper.get("publication") or {}
    if isinstance(pub, dict):
        if pub.get("venue"):
            lines.append(f"venue: {pub['venue']}")
        if pub.get("date_published") or pub.get("year"):
            lines.append(f"date: {pub.get('date_published') or pub.get('year')}")
    ids = paper.get("identifiers") or {}
    if isinstance(ids, dict) and ids.get("arxiv"):
        lines.append(f"arxiv: {ids['arxiv']}")
    content = paper.get("content") or {}
    if isinstance(content, dict):
        if content.get("abstract"):
            lines.append("")
            lines.append("abstract (from metadata):")
            lines.append(str(content["abstract"]))
        kws = content.get("keywords")
        if kws:
            lines.append(f"keywords: {kws}")
    meta_block = "\n".join(lines)
    return title, meta_block


def build_bundle_from_directory(directory: Path) -> PaperBundle:
    """Load arxiv2md-beta style directory: paper.yml + main md + optional refs/appendix."""
    directory = directory.resolve()
    yml_path = directory / "paper.yml"
    title = directory.name
    meta_text = ""

    if yml_path.is_file():
        title, meta_text = load_paper_yml_meta(yml_path)
    else:
        logger.warning(f"No paper.yml in {directory}; using directory name as title hint")

    main_path = _pick_main_md(directory)
    main_body = main_path.read_text(encoding="utf-8")
    if not title or title == directory.name:
        title = _first_heading_or_title(main_body, main_path.stem)

    sections, section_order, section_headings = split_markdown_ordered(main_body)
    extra_keys = [k for k in section_order if k.startswith("extra:")]
    if extra_keys:
        logger.info(f"Non-standard headings → generic prompt: {extra_keys[:12]}")

    refs_path = _find_sidecar(directory, "*-References.md")
    if refs_path and refs_path.is_file():
        ref_body = refs_path.read_text(encoding="utf-8").strip()
        if ref_body:
            sections["references"] = ref_body
            if "references" not in section_headings:
                section_headings["references"] = f"References（侧车 {refs_path.name}）"
            if "references" not in section_order:
                section_order.append("references")
            logger.info(f"References from sidecar: {refs_path}")

    apx_path = _find_sidecar(directory, "*-Appendix.md") or _find_sidecar(
        directory, "*-Appendices.md"
    )
    if apx_path and apx_path.is_file():
        apx_body = apx_path.read_text(encoding="utf-8").strip()
        if apx_body:
            sections["appendices"] = apx_body
            if "appendices" not in section_headings:
                section_headings["appendices"] = f"Appendices（侧车 {apx_path.name}）"
            if "appendices" not in section_order:
                section_order.append("appendices")
            logger.info(f"Appendices from sidecar: {apx_path}")

    parts = [main_body]
    if refs_path and refs_path.is_file():
        parts.append(refs_path.read_text(encoding="utf-8"))
    if apx_path and apx_path.is_file():
        parts.append(apx_path.read_text(encoding="utf-8"))
    full_text = "\n\n".join(parts)

    return PaperBundle(
        paper_title=title,
        meta_text=meta_text or f"(no paper.yml) title from content: {title}",
        sections=sections,
        full_text=full_text,
        source_description=str(directory),
        main_path=main_path,
        section_order=section_order,
        section_headings=section_headings,
    )


def build_bundle_from_file(md_path: Path) -> PaperBundle:
    """Load a single paper markdown file and split by headings."""
    md_path = md_path.resolve()
    text = md_path.read_text(encoding="utf-8")
    title = _first_heading_or_title(text, md_path.stem)
    sections, section_order, section_headings = split_markdown_ordered(text)
    extra_keys = [k for k in section_order if k.startswith("extra:")]
    if extra_keys:
        logger.info(f"Non-standard headings → generic prompt: {extra_keys[:12]}")
    return PaperBundle(
        paper_title=title,
        meta_text=f"Single-file input: {md_path.name}\n\n(title inferred from first heading or filename)",
        sections=sections,
        full_text=text,
        source_description=str(md_path),
        main_path=md_path,
        section_order=section_order,
        section_headings=section_headings,
    )


def resolve_prompt_path(prompt_dir: str, prompt_filename: str, project_root: Path | None = None) -> Path:
    """Resolve @prompts/paper/foo.md, explicit path, repo ``prompts/paper/``, or package symlink."""
    base = prompt_dir.strip()
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
        candidates.append((root / rel / prompt_filename).resolve())
    else:
        candidates.append((Path(base).expanduser().resolve() / prompt_filename).resolve())

    import ask_llm as _ask_llm

    pkg_root = Path(_ask_llm.__file__).resolve().parent
    # src layout: .../src/ask_llm/ -> repo root is parent of ``src``
    repo_prompts = (pkg_root.parent.parent / "prompts" / "paper" / prompt_filename).resolve()
    candidates.append(repo_prompts)
    # Package ``ask_llm/prompts`` (often a symlink to repo ``prompts/`` for setuptools)
    pkg_prompts = (pkg_root / "prompts" / "paper" / prompt_filename).resolve()
    candidates.append(pkg_prompts)

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Paper prompt not found: {prompt_filename} (tried: {', '.join(str(p) for p in candidates)})"
    )


def resolve_prompt_key(key: str) -> str:
    """Map job key (e.g. ``extra:foo``) to template registry key."""
    if key.startswith("extra:"):
        return "generic"
    if key.startswith("appendices:h2:"):
        return "appendices"
    return key


def _unique_appendix_slug(title: str, used: set[str]) -> str:
    base = _slug_for_extra_heading(title)
    slug = base
    n = 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def _split_appendix_body_by_h2(text: str) -> list[tuple[str, str]]:
    """
    Split appendix Markdown by ``##`` (level-2) headings only.

    Lines before the first ``##`` are prepended to the first subsection body (context for the first H2).

    Returns:
        List of ``(h2_heading, body)``. If there is no ``##`` in the text, returns
        ``[("", full_text)]`` so the caller can treat it as a single unsplit block.
    """
    lines = text.splitlines()
    h2_re = re.compile(r"^##\s+(.+?)\s*$")
    blocks: list[tuple[str, list[str]]] = []
    cur_title: str | None = None
    cur_lines: list[str] = []
    leading: list[str] = []

    for line in lines:
        m = h2_re.match(line)
        if m:
            if cur_title is None:
                cur_title = m.group(1).strip()
                cur_lines = leading[:]
                leading = []
            else:
                blocks.append((cur_title, cur_lines))
                cur_title = m.group(1).strip()
                cur_lines = []
        else:
            if cur_title is None:
                leading.append(line)
            else:
                cur_lines.append(line)

    if cur_title is not None:
        blocks.append((cur_title, cur_lines))

    if not blocks:
        joined = "\n".join(leading).strip()
        return [("", joined)] if joined else []

    out: list[tuple[str, str]] = []
    for h, ls in blocks:
        body = "\n".join(ls).strip()
        out.append((h, body))
    return out


def expand_appendices_into_h2_jobs(body: str) -> list[tuple[str, str, str]]:
    """
    Split large appendix content by ``##`` (second-level headings) into separate explain jobs.

    Returns list of ``(job_key, section_body, h2_heading)`` where ``h2_heading`` is the original
    ``##`` title (for prompts/display). Empty ``h2_heading`` means a single unsplit ``appendices`` job.

    - No ``##`` in body: one job ``("appendices", body, "")``.
    - One or more ``##``: jobs ``("appendices:h2:<slug>", body, h2_heading)`` per subsection.
    """
    raw = (body or "").strip()
    if not raw:
        return []

    parts = _split_appendix_body_by_h2(raw)
    if not parts:
        return []

    if len(parts) == 1 and not parts[0][0]:
        only_body = parts[0][1]
        if not only_body.strip():
            return []
        return [("appendices", only_body, "")]

    used_slugs: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for h2_title, sec_body in parts:
        if not h2_title:
            continue
        sec_body = sec_body.strip()
        if not sec_body:
            logger.warning(f"Skipping empty appendix subsection under ## {h2_title!r}")
            continue
        slug = _unique_appendix_slug(h2_title, used_slugs)
        key = f"appendices:h2:{slug}"
        out.append((key, sec_body, h2_title))
    if not out and parts:
        logger.warning("All appendix ## subsections were empty; using one unsplit appendices job")
        return [("appendices", raw, "")]
    return out


def load_prompt_template(prompt_dir: str, key: str, project_root: Path | None = None) -> str:
    """Load prompt file for meta | section key | full | generic (via ``extra:*``)."""
    pk = resolve_prompt_key(key)
    fname = _PROMPT_FILES.get(pk)
    if not fname:
        raise KeyError(f"Unknown prompt key: {key}")
    path = resolve_prompt_path(prompt_dir, fname, project_root=project_root)
    return path.read_text(encoding="utf-8").strip()


def format_prompt(
    template: str,
    paper_title: str,
    section_name: str,
    content: str,
    section_heading: str | None = None,
) -> str:
    """Fill template placeholders: title, section_name, section_heading, content."""
    sh = section_heading if section_heading is not None else section_name
    return (
        template.replace("{paper_title}", paper_title)
        .replace("{section_name}", section_name)
        .replace("{section_heading}", sh)
        .replace("{content}", content)
    )


def section_display_name(bundle: PaperBundle, key: str) -> str:
    """Human-readable section label for prompts (ZH + original heading for extras)."""
    if key.startswith("extra:"):
        return bundle.section_headings.get(key, key)
    return SECTION_LABELS_ZH.get(key, key)


def section_label_for_job(
    bundle: PaperBundle, key: str, appendix_h2_heading: str | None = None
) -> str:
    """Like ``section_display_name`` but includes appendix ``##`` subsection title when split."""
    if appendix_h2_heading and key.startswith("appendices:h2:"):
        return f"{SECTION_LABELS_ZH['appendices']} — {appendix_h2_heading}"
    return section_display_name(bundle, key)


def prompt_template_summary(template: str, max_len: int = 220) -> str:
    """First non-comment line of template as a short summary for file preambles."""
    for line in template.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:max_len]
    return template[:max_len].strip()


def explain_source_line(bundle: PaperBundle, key: str) -> str:
    """One-line Chinese description of which part of the paper the job uses."""
    if key == "meta":
        return "来自 `paper.yml` 等元数据（目录模式）或单文件输入说明；非正文切片。"
    if key == "full":
        main = bundle.main_path.name if bundle.main_path else "主 Markdown"
        return f"论文全文拼接（主文件：{main}；若存在侧车则含参考文献与附录）。"
    raw = bundle.section_headings.get(key, "")
    if key.startswith("extra:"):
        return (
            f"原论文 Markdown 中标题为「{raw}」的小节（非标准章节名，使用通用解析模板）。"
        )
    label = SECTION_LABELS_ZH.get(key, key)
    if raw:
        return f"原论文 Markdown 中标题为「{raw}」的小节（对应标准章节：{label}）。"
    return f"原论文对应小节（{label}）。"


def build_explain_preamble_text(
    bundle: PaperBundle,
    key: str,
    prompt_key: str,
    template_text: str,
    source_override: str | None = None,
) -> str:
    """Markdown block prepended to each ``*.explain.md`` output."""
    fname = _PROMPT_FILES.get(prompt_key, f"{prompt_key}.md")
    relpath = f"prompts/paper/{fname}"
    summary = prompt_template_summary(template_text)
    src = source_override if source_override is not None else explain_source_line(bundle, key)
    return (
        "## 说明\n\n"
        f"- **来源**：{src}\n"
        f"- **解析模板**：`{relpath}`\n"
        f"- **提示摘要**：{summary}\n\n"
        "---\n\n"
    )


def _strip_markdown_code_fence(text: str) -> str:
    """Remove a single leading/trailing ``` fence (optionally ```json)."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if not lines:
        return s
    if lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_outer_json_object(text: str) -> str | None:
    """If text contains a JSON object, return substring from first ``{`` to last ``}``."""
    t = text.strip()
    if not t:
        return None
    i = t.find("{")
    j = t.rfind("}")
    if i == -1 or j <= i:
        return None
    return t[i : j + 1]


def _json_object_to_markdown(obj: dict, *, level: int = 2) -> str:
    """Turn a dict (e.g. model JSON output) into Markdown headings and body text."""
    lines: list[str] = []
    h_level = max(2, min(level, 6))
    for k, v in obj.items():
        title = str(k).strip()
        lines.append(f"{'#' * h_level} {title}\n")
        if isinstance(v, dict):
            lines.append(_json_object_to_markdown(v, level=h_level + 1))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    lines.append(_json_object_to_markdown(item, level=h_level + 1))
                else:
                    lines.append(f"- {item}\n")
            lines.append("")
        else:
            lines.append(f"{v}\n\n")
    return "".join(lines)


def normalize_paper_explain_response(text: str) -> str:
    """
    Convert a sole JSON object response into Markdown when models ignore Markdown instructions.

    Some reasoning models still emit structured JSON despite prompts asking for Markdown; this
    keeps explain/*.md readable in viewers. Unchanged if not parseable as JSON or not object-shaped.
    """
    original = text or ""
    stripped = original.strip()
    if not stripped:
        return original
    candidate = _strip_markdown_code_fence(stripped)
    blob = candidate if candidate.startswith("{") else _extract_outer_json_object(candidate)
    if not blob:
        return original
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return original
    if isinstance(data, dict) and data:
        return _json_object_to_markdown(data).strip()
    if isinstance(data, list) and data:
        parts: list[str] = []
        for item in data:
            if isinstance(item, dict):
                parts.append(_json_object_to_markdown(item, level=2))
            else:
                parts.append(f"- {item}")
        return "\n\n".join(parts).strip()
    return original


def explain_output_filename(index: int, key: str) -> str:
    """Target filename under explain/ with stable numeric prefix (document order)."""
    if key == "meta":
        return f"{index}-meta.explain.md"
    if key == "full":
        return f"{index}-full.explain.md"
    if key.startswith("appendices:h2:"):
        slug = key.split(":", 2)[2]
        return f"d-appendices-{slug}.explain.md"
    if key.startswith("extra:"):
        slug = key.split(":", 1)[1]
        return f"{index}-{slug}.explain.md"
    return f"{index}-{key}.explain.md"
