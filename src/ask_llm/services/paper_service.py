"""Paper explanation orchestration service.

Moves the core paper-explain workflow (load bundle, build jobs, run LLM calls,
write outputs) out of the CLI command so the command module stays focused on
argument parsing and user-facing error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ask_llm.config.manager import ConfigManager
from ask_llm.config.paper_explain_pipeline import (
    PaperExplainPipelineConfig,
    load_paper_explain_pipeline,
    parse_section_job_key,
)
from ask_llm.config.unified_config import UnifiedConfig
from ask_llm.core.batch import BatchResult, BatchStatistics, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.execution_report import ExecutionReport, build_report_from_batch_results
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.models import AppConfig
from ask_llm.core.paper_explain import (
    PaperBundle,
    build_bundle_from_directory,
    build_bundle_from_file,
    build_combo_section_body,
    build_explain_preamble_text,
    expand_appendices_into_h2_jobs,
    explain_output_filename,
    format_prompt,
    load_prompt_template,
    normalize_paper_explain_response,
    resolve_prompt_key,
    section_label_for_job,
)
from ask_llm.core.tasks.builders import build_paper_explain_task
from ask_llm.utils.console import console
from ask_llm.utils.file_handler import FileHandler
from ask_llm.utils.pricing import format_cost_estimate
from ask_llm.utils.provider_router import build_fallback_chain
from ask_llm.utils.provider_specs import (
    load_providers_model_limits,
    resolve_paper_max_tokens,
)
from ask_llm.utils.token_counter import TokenCounter

PricingMap = dict[tuple[str, str], dict[str, float]]


@dataclass
class PaperExplainOptions:
    """Resolved paper-explain options used by PaperService."""

    run_mode: str
    section_filter: set[str] | None
    temperature: float | None
    force: bool
    include_metadata: bool
    concurrency: int
    dry_run: bool
    resume: bool
    pipeline_path: str | None
    use_fallback: bool = True


@dataclass
class PaperJobResult:
    """Result of a single paper-explain job."""

    task_id: int
    key: str
    output_file: Path
    success: bool
    error: str | None = None


@dataclass
class PaperSessionResult:
    """Aggregate result of a paper-explain session."""

    job_results: list[PaperJobResult] = field(default_factory=list)
    statistics: dict[str, BatchStatistics] = field(default_factory=dict)
    total_jobs: int = 0
    skipped_count: int = 0
    explain_root: Path | None = None
    report: ExecutionReport | None = None


class PaperService:
    """High-level service for explaining papers via LLM APIs."""

    def __init__(
        self,
        config_manager: ConfigManager,
        unified_config: UnifiedConfig,
        *,
        provider: str,
        model: str,
        pricing_map: PricingMap,
        pricing_source: Path | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        """Initialize the paper-explain service.

        Args:
            config_manager: Active config manager (provider/model already resolved).
            unified_config: Loaded unified configuration.
            provider: Resolved provider name.
            model: Resolved model name.
            pricing_map: Pricing data for cost estimates.
            pricing_source: Optional path/label of the pricing source.
            app_config: Loaded application config; required for provider fallback chain.
        """
        self.config_manager = config_manager
        self.unified_config = unified_config
        self.provider = provider
        self.model = model
        self.pricing_map = pricing_map
        self.pricing_source = pricing_source
        self.app_config = app_config
        self._last_results: list[BatchResult] | None = None

    def explain_paper(
        self,
        input_path: Path,
        options: PaperExplainOptions,
    ) -> PaperSessionResult:
        """Explain a paper file or directory.

        Args:
            input_path: Resolved path to a .md file or arxiv2md-beta directory.
            options: Resolved paper-explain options.

        Returns:
            PaperSessionResult aggregating per-job outcomes and statistics.

        Raises:
            ValueError: If run_mode is invalid or input is not a file/directory.
            FileNotFoundError: If required files are missing.
            typer.Exit: On dry-run completion or when no jobs remain.
        """
        import typer

        run_norm = options.run_mode.strip().lower()
        if run_norm not in ("sections", "full", "all"):
            raise ValueError("--run must be one of: sections, full, all")

        paper_cfg = self.unified_config.paper
        prompt_dir = paper_cfg.prompt_dir
        pipeline_yaml = (options.pipeline_path or "").strip() or paper_cfg.pipeline_config
        explain_pipeline = load_paper_explain_pipeline(pipeline_yaml)
        console.print_info(f"Paper pipeline config: {pipeline_yaml}")
        out_sub = paper_cfg.output_subdir.strip() or "explain"

        if input_path.is_file():
            if input_path.suffix.lower() not in (".md", ".markdown"):
                raise ValueError("Paper file must be .md or .markdown")
            bundle = build_bundle_from_file(input_path, pipeline=explain_pipeline)
            explain_root = input_path.parent / out_sub
        elif input_path.is_dir():
            bundle = build_bundle_from_directory(input_path, pipeline=explain_pipeline)
            explain_root = input_path / out_sub
        else:
            raise ValueError(f"Not a file or directory: {input_path}")

        explain_root.mkdir(parents=True, exist_ok=True)
        console.print_info(f"Output directory: {explain_root}")
        console.print_info(f"Paper title: {bundle.paper_title}")

        jobs = self._build_jobs(bundle, explain_pipeline, run_norm, options.section_filter)
        if not jobs:
            raise ValueError("No jobs to run (check --run and --sections, or empty sections)")

        from ask_llm.core.constants import DEFAULT_FALLBACK_MODEL

        paper_max_tokens = paper_cfg.max_output_tokens
        full_model_name = (paper_cfg.full_model or "").strip() or DEFAULT_FALLBACK_MODEL
        model_limits_map, _ = load_providers_model_limits()
        section_job_model = self.model
        current_provider = self.config_manager.current_provider_name

        if options.dry_run:
            self._dry_run(
                jobs,
                bundle,
                explain_pipeline,
                prompt_dir,
                full_model_name,
                section_job_model,
                current_provider,
            )
            raise typer.Exit(0)

        jobs_with_orig_idx, skipped_count = self._apply_resume_or_force(
            jobs,
            explain_root,
            explain_pipeline,
            options.resume,
            options.force,
        )

        if not jobs_with_orig_idx:
            console.print_info("All sections already completed. Nothing to do.")
            raise typer.Exit(0)

        idx_to_meta: dict[int, tuple[str, str, str | None]] = {}
        paper_tasks: list[BatchTask] = []

        for orig_idx, (key, body, appendix_h2) in jobs_with_orig_idx:
            template = load_prompt_template(prompt_dir, key, pipeline=explain_pipeline)
            label = section_label_for_job(bundle, key, appendix_h2, pipeline=explain_pipeline)
            heading = self._resolve_heading(bundle, key, appendix_h2, explain_pipeline)
            full_prompt = format_prompt(
                template,
                paper_title=bundle.paper_title,
                section_name=label,
                content=body,
                section_heading=heading,
            )
            job_model = full_model_name if key.startswith("full") else section_job_model
            eff_max = resolve_paper_max_tokens(job_model, paper_max_tokens, model_limits_map)
            logger.debug(
                f"paper job: key={key!r} model={job_model!r} max_tokens={eff_max} "
                f"(paper.max_output_tokens={paper_max_tokens})"
            )
            model_config = ModelConfig(
                provider=current_provider,
                model=job_model,
                temperature=options.temperature,
                max_tokens=eff_max,
            )
            fallback_configs: list[ModelConfig] = []
            if options.use_fallback and self.app_config is not None:
                fallback_configs = build_fallback_chain(self.app_config, model_config)
            idx_to_meta[orig_idx] = (key, template, appendix_h2)
            paper_tasks.append(
                build_paper_explain_task(
                    orig_idx,
                    full_prompt,
                    model_settings=model_config,
                    output_filename=f"paper:{key}",
                    return_reasoning=(key.startswith("full")),
                    fallback_model_configs=fallback_configs,
                )
            )

        max_workers = max(1, min(options.concurrency, len(paper_tasks)))
        console.print_info(
            f"Paper explain: {len(paper_tasks)} job(s), "
            f"up to {max_workers} concurrent worker(s) (GlobalBatchProcessor, same as trans)"
        )
        results, processor = run_global_batch_tasks(
            paper_tasks,
            self.config_manager,
            max_workers=options.concurrency,
            max_retries=3,
            show_progress=True,
            clamp_workers_to_task_count=True,
        )
        self._last_results = list(results)

        failed = [r for r in results if r.status != TaskStatus.SUCCESS]
        if failed:
            for r in failed:
                console.print_error(f"Paper job {r.task_id} failed: {r.error or 'unknown error'}")
            raise typer.Exit(1)

        session_result = PaperSessionResult(
            total_jobs=len(jobs),
            skipped_count=skipped_count,
            explain_root=explain_root,
        )
        for result in sorted(results, key=lambda r: r.task_id):
            job_result = self._write_result(
                result,
                idx_to_meta,
                bundle,
                explain_pipeline,
                explain_root,
                prompt_dir,
                full_model_name,
                options.include_metadata,
                options.force,
            )
            session_result.job_results.append(job_result)

        session_result.statistics = processor.calculate_statistics(results)
        session_result.report = build_report_from_batch_results(
            "paper",
            results,
            metadata={
                "input_path": str(input_path),
                "run_mode": options.run_mode,
                "pipeline": options.pipeline_path,
            },
        )
        self._print_usage(session_result.statistics)
        return session_result

    def _build_jobs(
        self,
        bundle: PaperBundle,
        explain_pipeline: PaperExplainPipelineConfig,
        run_norm: str,
        section_filter: set[str] | None,
    ) -> list[tuple[str, str, str | None]]:
        """Build the list of explain jobs from the paper bundle."""
        jobs: list[tuple[str, str, str | None]] = []
        combo_consumed = explain_pipeline.combo_consumed_keys()

        if run_norm in ("sections", "all"):
            if self._want("meta", section_filter, explain_pipeline):
                jobs.append(("meta", bundle.meta_text, None))
            for key in bundle.section_order:
                if not self._want(key, section_filter, explain_pipeline):
                    continue
                if key in combo_consumed and (
                    section_filter is None or key.lower() not in section_filter
                ):
                    continue
                body = bundle.sections.get(key)
                if not body or not str(body).strip():
                    logger.warning(f"Skipping empty section: {key}")
                    continue
                if key == "appendices":
                    apx_jobs = expand_appendices_into_h2_jobs(body)
                    if not apx_jobs:
                        logger.warning("Skipping empty appendix expansion")
                        continue
                    n_h2 = sum(1 for jk, _, _ in apx_jobs if jk.startswith("appendices:h2:"))
                    if n_h2 >= 1:
                        console.print_info(
                            f"Appendices: {n_h2} explain job(s) split by ## → "
                            f"d-appendices-<slug>.explain.md"
                        )
                    for jk, jbody, h2 in apx_jobs:
                        jobs.append((jk, jbody, h2 or None))
                else:
                    entries = explain_pipeline.resolved_section_prompts(key)
                    if len(entries) == 1:
                        jobs.append((key, body, None))
                    else:
                        for e in entries:
                            stem = Path(e.file).stem
                            jobs.append((f"{key}:{stem}", body, None))

            if explain_pipeline.section_combos:
                for combo in explain_pipeline.section_combos:
                    merged = build_combo_section_body(bundle, combo.keys)
                    if not merged:
                        logger.warning(f"Skipping empty section combo: {combo.id}")
                        continue
                    for e in combo.prompts:
                        stem = Path(e.file).stem
                        jk = f"combo:{combo.id}:{stem}"
                        if not self._want(jk, section_filter, explain_pipeline):
                            continue
                        jobs.append((jk, merged, None))

        if run_norm in ("full", "all") and self._want("full", section_filter, explain_pipeline):
            ft = bundle.full_text.strip()
            if not ft:
                raise ValueError("Full text is empty; cannot run full analysis")
            full_entries = explain_pipeline.resolved_full_prompts()
            if len(full_entries) > 1:
                console.print_info(
                    "Full paper: "
                    f"{len(full_entries)} separate LLM call(s) on the same body — "
                    + ", ".join(e.file for e in full_entries)
                )
                for fp in full_entries:
                    stem = Path(fp.file).stem
                    jobs.append((f"full:{stem}", ft, None))
            else:
                jobs.append(("full", ft, None))

        return jobs

    def _want(
        self,
        key: str,
        section_filter: set[str] | None,
        explain_pipeline: PaperExplainPipelineConfig,
    ) -> bool:
        """Return True if the job key passes the --sections filter."""
        if section_filter is None:
            return True
        lk = key.lower()
        for f in section_filter:
            fl = f.lower()
            if fl == lk:
                return True
            if fl == "full" and lk.startswith("full:"):
                return True
            if fl == "combo" and lk.startswith("combo:"):
                return True
            if (
                lk.startswith("combo:")
                and fl.startswith("combo:")
                and (lk == fl or lk.startswith(fl + ":"))
            ):
                return True
            base, stem = parse_section_job_key(key, explain_pipeline)
            if base is not None and stem is not None:
                if fl == base:
                    return True
                if fl == f"{base}:{stem}":
                    return True
        return False

    def _resolve_heading(
        self,
        bundle: PaperBundle,
        key: str,
        appendix_h2: str | None,
        explain_pipeline: PaperExplainPipelineConfig,
    ) -> str | None:
        """Resolve the human-readable heading for a job key."""
        if key.startswith("combo:"):
            parts = key.split(":")
            cid = parts[1] if len(parts) >= 2 else ""
            combo = explain_pipeline.combo_by_id(cid)
            if combo:
                hp = [bundle.section_headings.get(k, "") for k in combo.keys]
                return " + ".join(h for h in hp if h)
            return bundle.section_headings.get(key)
        if appendix_h2 and key.startswith("appendices:h2:"):
            return appendix_h2
        base, _ = parse_section_job_key(key, explain_pipeline)
        return bundle.section_headings.get(base or key)

    def _dry_run(
        self,
        jobs: list[tuple[str, str, str | None]],
        bundle: PaperBundle,
        explain_pipeline: PaperExplainPipelineConfig,
        prompt_dir: str,
        full_model_name: str,
        section_job_model: str,
        current_provider: str,
    ) -> None:
        """Print a preview of planned jobs and token estimates without calling APIs."""
        console.print(f"\n[bold]Dry Run — {len(jobs)} job(s) planned:[/bold]")
        total_tokens = 0
        for idx, (key, body, appendix_h2) in enumerate(jobs):
            template = load_prompt_template(prompt_dir, key, pipeline=explain_pipeline)
            label = section_label_for_job(bundle, key, appendix_h2, pipeline=explain_pipeline)
            heading = self._resolve_heading(bundle, key, appendix_h2, explain_pipeline)
            full_prompt = format_prompt(
                template,
                paper_title=bundle.paper_title,
                section_name=label,
                content=body,
                section_heading=heading,
            )
            job_model = full_model_name if key.startswith("full") else section_job_model
            tok = TokenCounter.count_tokens(full_prompt, job_model)
            total_tokens += tok
            console.print(f"  [{idx:2}] {key:<30} {tok:>6} tokens")
        console.print(f"\n  Total estimated input: {total_tokens:,} tokens across {len(jobs)} jobs")
        if self.pricing_map:
            console.print(
                format_cost_estimate(
                    current_provider,
                    section_job_model,
                    total_tokens,
                    total_tokens * 3,
                    self.pricing_map,
                    pricing_source=self.pricing_source,
                )
            )

    def _apply_resume_or_force(
        self,
        jobs: list[tuple[str, str, str | None]],
        explain_root: Path,
        explain_pipeline: PaperExplainPipelineConfig,
        resume: bool,
        force: bool,
    ) -> tuple[list[tuple[int, tuple[str, str, str | None]]], int]:
        """Filter jobs by --resume or guard against existing outputs.

        Returns:
            (jobs_with_original_index, skipped_count)
        """
        if resume:
            original_jobs = list(enumerate(jobs))
            filtered = [
                (orig_idx, job)
                for orig_idx, job in original_jobs
                if not (
                    explain_root / explain_output_filename(orig_idx, job[0], explain_pipeline)
                ).exists()
            ]
            skipped = len(original_jobs) - len(filtered)
            if skipped:
                console.print_info(f"--resume: skipping {skipped} already-completed section(s)")
            return filtered, skipped

        for idx, (key, _, _) in enumerate(jobs):
            out_name = explain_output_filename(idx, key, explain_pipeline)
            out_file = explain_root / out_name
            if out_file.exists() and not force:
                raise FileExistsError(f"Output exists: {out_file}. Use --force or --resume.")
        return list(enumerate(jobs)), 0

    def _write_result(
        self,
        result: BatchResult,
        idx_to_meta: dict[int, tuple[str, str, str | None]],
        bundle: PaperBundle,
        explain_pipeline: PaperExplainPipelineConfig,
        explain_root: Path,
        prompt_dir: str,
        full_model_name: str,
        include_metadata: bool,
        force: bool,
    ) -> PaperJobResult:
        """Write a single paper-explain result to disk."""
        idx = result.task_id
        key, template, appendix_h2 = idx_to_meta[idx]
        pk = resolve_prompt_key(key, explain_pipeline)
        out_name = explain_output_filename(idx, key, explain_pipeline)
        out_file = explain_root / out_name
        main_name = bundle.main_path.name if bundle.main_path else "主 Markdown"

        if key == "full" or key.startswith("full:"):
            src_full = (
                f"论文全文拼接（主文件：{main_name}；若存在侧车则含参考文献与附录）。"
                f"全文解读使用模型 `{full_model_name}`，并写入 API 返回的推理内容（若存在）。"
            )
            preamble = build_explain_preamble_text(
                bundle,
                key,
                pk,
                template,
                source_override=src_full,
                pipeline=explain_pipeline,
                prompt_dir=prompt_dir,
            )
        elif key.startswith("appendices:h2:") and appendix_h2:
            src_apx = (
                f"附录（侧车或正文中的 Appendices）按二级标题「##」切分后的「{appendix_h2}」小节。"
            )
            preamble = build_explain_preamble_text(
                bundle,
                key,
                pk,
                template,
                source_override=src_apx,
                pipeline=explain_pipeline,
                prompt_dir=prompt_dir,
            )
        else:
            preamble = build_explain_preamble_text(
                bundle,
                key,
                pk,
                template,
                pipeline=explain_pipeline,
                prompt_dir=prompt_dir,
            )

        body_out = normalize_paper_explain_response(result.response or "")
        if result.reasoning:
            body_out = (
                "## 推理过程（思维链）\n\n"
                + result.reasoning.strip()
                + "\n\n---\n\n## 正文解析\n\n"
                + body_out
            )
        if include_metadata and result.metadata:
            body_out = result.metadata.format() + body_out
        text_out = preamble + body_out
        FileHandler.write(str(out_file), text_out, force=force)
        console.print_success(f"Wrote {out_file}")
        return PaperJobResult(
            task_id=idx,
            key=key,
            output_file=out_file,
            success=True,
        )

    def _print_usage(self, statistics: dict[str, BatchStatistics]) -> None:
        """Print aggregate token/cost usage by model."""
        console.print()
        if len(statistics) > 1:
            console.print("[bold]Paper explain — usage by model[/bold]")
        for model_key in sorted(statistics.keys()):
            st = statistics[model_key]
            parts = model_key.split("/", 1)
            prov, mod = parts[0], parts[1] if len(parts) > 1 else ""
            if len(statistics) > 1:
                console.print(f"[bold]{model_key}[/bold]")
            console.print(
                format_cost_estimate(
                    prov,
                    mod,
                    st.total_input_tokens,
                    st.total_output_tokens,
                    self.pricing_map,
                    pricing_source=self.pricing_source,
                )
            )

    def export_report(self, report_path: str | None) -> str | None:
        """Export the execution report to ``report_path`` if available.

        Args:
            report_path: Destination path for the JSON report.

        Returns:
            The exported path, or ``None`` if no report was generated or no path
            was requested.
        """
        if not report_path or self._last_results is None:
            return None
        report = build_report_from_batch_results(
            "paper",
            self._last_results,
            metadata={"provider": self.provider, "model": self.model},
        )
        report.to_json_file(report_path)
        console.print_info(f"Execution report saved to: {report_path}")
        return report_path
