"""Typer command `paper` (split from former cli.py)."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger
from typing_extensions import Annotated

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.config.cli_session import (
    apply_cli_overrides_and_gate_api_key,
    load_cli_session,
    resolve_default_model_or_exit,
)
from ask_llm.core.batch import (
    BatchTask,
    ModelConfig,
    TaskStatus,
)
from ask_llm.core.global_batch_runner import run_global_batch_tasks
from ask_llm.core.paper_explain import (
    build_bundle_from_directory,
    build_bundle_from_file,
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
from ask_llm.utils.pricing import format_cost_estimate, load_providers_pricing
from ask_llm.utils.provider_specs import (
    load_providers_model_limits,
    resolve_paper_max_tokens,
)

try:
    import llm_engine  # noqa: F401 — fail fast if engine missing
except ImportError:
    console.print_error(
        "llm_engine is required but not installed. Please install it with: pip install llm-engine"
    )
    raise


def paper(
    input_path: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Path to a paper .md file or an arxiv2md-beta output directory",
        ),
    ],
    run_mode: Annotated[
        str,
        typer.Option(
            "--run",
            "-r",
            help="sections (per-section + meta), full (whole paper), or all",
        ),
    ] = "all",
    sections: Annotated[
        str | None,
        typer.Option(
            "--sections",
            "-s",
            help="Comma-separated keys: meta,abstract,...,appendices (split by ## into d-appendices-*.explain.md),full",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-a", help="API provider"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", "-t", help="Sampling temperature", min=0.0, max=2.0),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Configuration file path"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing explain/*.md files"),
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option("--metadata", help="Include token/latency metadata in each output file"),
    ] = False,
    skip_api_key_check: Annotated[
        bool,
        typer.Option("--skip-api-key-check", help="Skip API key check"),
    ] = False,
    providers_pricing: Annotated[
        str | None,
        typer.Option(
            "--providers-pricing",
            help="Path to providers.yml (pricing_per_million_tokens). "
            "Default search: ASK_LLM_PROVIDERS_YML, ./providers.yml, package root, ~/.config/ask_llm/providers.yml",
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-j",
            help="Parallel LLM calls (default: paper.concurrency in config; thread pool for I/O)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show detected sections and token estimates without making API calls",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip sections whose output already exists",
        ),
    ] = False,
) -> None:
    """
    Explain a paper: split Markdown by headings (or load arxiv2md-beta dir), call LLM per section,
    write results to ./explain/ next to the input file or directory.

    Multiple sections run in parallel via GlobalBatchProcessor (same pipeline as ``trans``) when concurrency > 1.

    Examples:
        ask-llm paper -i paper.md --run all
        ask-llm paper -i output2/20170612-Arxiv-Attention-Is-All-You-Need --run sections
        ask-llm paper -i paper.md -r full
        ask-llm paper -i paper.md -j 8
    """
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        console.print_error(f"Path not found: {path}")
        raise typer.Exit(1)

    run_norm = run_mode.strip().lower()
    if run_norm not in ("sections", "full", "all"):
        console.print_error("--run must be one of: sections, full, all")
        raise typer.Exit(1)

    section_filter: set[str] | None = None
    if sections:
        section_filter = {x.strip().lower() for x in sections.split(",") if x.strip()}

    try:
        load_result, config_manager = load_cli_session(config_path)
        pricing_map, pricing_source = load_providers_pricing(providers_pricing)
        if pricing_source:
            console.print_info(f"API pricing loaded from: {pricing_source}")
        else:
            console.print_info(
                "No providers.yml with pricing found; token counts will still be shown, "
                "cost estimate unavailable (add pricing_per_million_tokens or use --providers-pricing)"
            )
        apply_cli_overrides_and_gate_api_key(
            config_manager,
            provider=provider,
            model=model,
            temperature=temperature,
            skip_api_key_check=skip_api_key_check,
        )
        default_model = resolve_default_model_or_exit(config_manager)
        paper_cfg = load_result.unified_config.paper
        prompt_dir = paper_cfg.prompt_dir
        out_sub = paper_cfg.output_subdir.strip() or "explain"
        workers = concurrency if concurrency is not None else paper_cfg.concurrency
        if workers < 1 or workers > 64:
            console.print_error("--concurrency / paper.concurrency must be between 1 and 64")
            raise typer.Exit(1)

        if path.is_file():
            if path.suffix.lower() not in (".md", ".markdown"):
                console.print_error("Paper file must be .md or .markdown")
                raise typer.Exit(1)
            bundle = build_bundle_from_file(path)
            explain_root = path.parent / out_sub
        elif path.is_dir():
            bundle = build_bundle_from_directory(path)
            explain_root = path / out_sub
        else:
            console.print_error(f"Not a file or directory: {path}")
            raise typer.Exit(1)

        explain_root.mkdir(parents=True, exist_ok=True)
        console.print_info(f"Output directory: {explain_root}")
        console.print_info(f"Paper title: {bundle.paper_title}")

        jobs: list[tuple[str, str, str | None]] = []

        def want(key: str) -> bool:
            if section_filter is None:
                return True
            return key in section_filter

        if run_norm in ("sections", "all"):
            if want("meta"):
                jobs.append(("meta", bundle.meta_text, None))
            for key in bundle.section_order:
                if not want(key):
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
                    jobs.append((key, body, None))

        if run_norm in ("full", "all") and want("full"):
            ft = bundle.full_text.strip()
            if not ft:
                console.print_error("Full text is empty; cannot run full analysis")
                raise typer.Exit(1)
            jobs.append(("full", ft, None))

        if not jobs:
            console.print_error("No jobs to run (check --run and --sections, or empty sections)")
            raise typer.Exit(1)

        # Dry-run: preview sections and token estimates
        if dry_run:
            from ask_llm.utils.token_counter import TokenCounter

            console.print(f"\n[bold]Dry Run — {len(jobs)} job(s) planned:[/bold]")
            total_tokens = 0
            for idx, (key, body, appendix_h2) in enumerate(jobs):
                template = load_prompt_template(prompt_dir, key)
                label = section_label_for_job(bundle, key, appendix_h2)
                heading = (
                    appendix_h2
                    if appendix_h2 and key.startswith("appendices:h2:")
                    else bundle.section_headings.get(key)
                )
                full_prompt = format_prompt(
                    template,
                    paper_title=bundle.paper_title,
                    section_name=label,
                    content=body,
                    section_heading=heading,
                )
                job_model = full_model_name if key == "full" else section_job_model
                tok = TokenCounter.count_tokens(full_prompt, job_model)
                total_tokens += tok
                out_name = explain_output_filename(idx, key)
                out_file = explain_root / out_name
                status = "[yellow]exists[/yellow]" if out_file.exists() else "[green]new[/green]"
                console.print(f"  [{idx:2}] {key:<30} {tok:>6} tokens  {status}  → {out_name}")
            console.print(f"\n  Total estimated input: {total_tokens:,} tokens across {len(jobs)} jobs")
            if pricing_map:
                console.print(format_cost_estimate(current_provider, section_job_model,
                                                   total_tokens, total_tokens * 3,
                                                   pricing_map, pricing_source=pricing_source))
            raise typer.Exit(0)

        # Conflict guard with --resume support
        if resume:
            original_jobs = list(enumerate(jobs))  # preserve (original_idx, job_tuple)
            filtered = [(orig_idx, job) for orig_idx, job in original_jobs
                        if not (explain_root / explain_output_filename(orig_idx, job[0])).exists()]
            skipped = len(original_jobs) - len(filtered)
            if skipped:
                console.print_info(f"--resume: skipping {skipped} already-completed section(s)")
            if not filtered:
                console.print_info("All sections already completed. Nothing to do.")
                raise typer.Exit(0)
            # Replace jobs with filtered list; carry original indices forward
            jobs_with_orig_idx: list[tuple[int, tuple[str, str, str | None]]] = filtered
        else:
            for idx, (key, _, _) in enumerate(jobs):
                out_name = explain_output_filename(idx, key)
                out_file = explain_root / out_name
                if out_file.exists() and not force:
                    console.print_error(f"Output exists: {out_file}. Use --force or --resume.")
                    raise typer.Exit(1)
            jobs_with_orig_idx = list(enumerate(jobs))

        paper_max_tokens = paper_cfg.max_output_tokens
        full_model_name = (paper_cfg.full_model or "").strip() or "deepseek-reasoner"
        model_limits_map, _providers_spec_path = load_providers_model_limits()
        # Section/meta jobs use CLI --model if set; otherwise the provider default (must not be
        # None or resolve_paper_max_tokens skips per-model caps and sends paper.max_output_tokens raw).
        section_job_model = (model or default_model).strip()
        current_provider = config_manager.current_provider_name

        idx_to_meta: dict[int, tuple[str, str, str | None]] = {}
        paper_tasks: list[BatchTask] = []

        for orig_idx, (key, body, appendix_h2) in jobs_with_orig_idx:
            template = load_prompt_template(prompt_dir, key)
            label = section_label_for_job(bundle, key, appendix_h2)
            heading = (
                appendix_h2
                if appendix_h2 and key.startswith("appendices:h2:")
                else bundle.section_headings.get(key)
            )
            full_prompt = format_prompt(
                template,
                paper_title=bundle.paper_title,
                section_name=label,
                content=body,
                section_heading=heading,
            )
            job_model = full_model_name if key == "full" else section_job_model
            eff_max = resolve_paper_max_tokens(job_model, paper_max_tokens, model_limits_map)
            logger.debug(
                f"paper job: key={key!r} model={job_model!r} max_tokens={eff_max} "
                f"(paper.max_output_tokens={paper_max_tokens})"
            )
            idx_to_meta[orig_idx] = (key, template, appendix_h2)
            paper_tasks.append(
                build_paper_explain_task(
                    orig_idx,
                    full_prompt,
                    task_model_config=ModelConfig(
                        provider=current_provider,
                        model=job_model,
                        temperature=temperature,
                        max_tokens=eff_max,
                    ),
                    output_filename=f"paper:{key}",
                    return_reasoning=(key == "full"),
                )
            )

        max_workers = max(1, min(workers, len(paper_tasks)))
        console.print_info(
            f"Paper explain: {len(paper_tasks)} job(s), "
            f"up to {max_workers} concurrent worker(s) (GlobalBatchProcessor, same as trans)"
        )
        results, processor = run_global_batch_tasks(
            paper_tasks,
            config_manager,
            max_workers=workers,
            max_retries=3,
            show_progress=True,
            clamp_workers_to_task_count=True,
        )

        failed = [r for r in results if r.status != TaskStatus.SUCCESS]
        if failed:
            for r in failed:
                console.print_error(f"Paper job {r.task_id} failed: {r.error or 'unknown error'}")
            raise typer.Exit(1)

        for result in sorted(results, key=lambda r: r.task_id):
            idx = result.task_id
            key, template, appendix_h2 = idx_to_meta[idx]
            pk = resolve_prompt_key(key)
            out_name = explain_output_filename(idx, key)
            out_file = explain_root / out_name
            main_name = bundle.main_path.name if bundle.main_path else "主 Markdown"
            if key == "full":
                src_full = (
                    f"论文全文拼接（主文件：{main_name}；若存在侧车则含参考文献与附录）。"
                    f"全文解读使用模型 `{full_model_name}`，并写入 API 返回的推理内容（若存在）。"
                )
                preamble = build_explain_preamble_text(
                    bundle, key, pk, template, source_override=src_full
                )
            elif key.startswith("appendices:h2:") and appendix_h2:
                src_apx = (
                    f"附录（侧车或正文中的 Appendices）按二级标题「##」切分后的「{appendix_h2}」小节。"
                )
                preamble = build_explain_preamble_text(
                    bundle, key, pk, template, source_override=src_apx
                )
            else:
                preamble = build_explain_preamble_text(bundle, key, pk, template)
            body_out = normalize_paper_explain_response(result.response or "")
            if result.reasoning:
                body_out = (
                    "## 推理过程（思维链）\n\n"
                    + result.reasoning.strip()
                    + "\n\n---\n\n## 正文解析\n\n"
                    + body_out
                )
            if metadata and result.metadata:
                body_out = result.metadata.format() + body_out
            text_out = preamble + body_out
            FileHandler.write(str(out_file), text_out, force=force)
            console.print_success(f"Wrote {out_file}")

        by_model = processor.calculate_statistics(results)
        console.print()
        if len(by_model) > 1:
            console.print("[bold]Paper explain — usage by model[/bold]")
        for model_key in sorted(by_model.keys()):
            st = by_model[model_key]
            parts = model_key.split("/", 1)
            prov, mod = parts[0], parts[1] if len(parts) > 1 else ""
            if len(by_model) > 1:
                console.print(f"[bold]{model_key}[/bold]")
            console.print(
                format_cost_estimate(
                    prov,
                    mod,
                    st.total_input_tokens,
                    st.total_output_tokens,
                    pricing_map,
                    pricing_source=pricing_source,
                )
            )

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except RuntimeError as e:
        console.print_error(f"API error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        raise_unexpected_cli_error("paper", e)
