"""Typer command `diagnose` — inspect execution reports produced by batch/trans/paper."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.errors import cli_errors
from ask_llm.config.cli_session import load_pricing_with_hint
from ask_llm.core.batch_models import TaskStatus
from ask_llm.core.execution_report import ExecutionReport
from ask_llm.core.telemetry import ErrorCategory
from ask_llm.utils.console import console
from ask_llm.utils.pricing import estimate_cost_cny, lookup_pricing


def diagnose(
    report_path: Annotated[
        str,
        typer.Argument(help="Path to the JSON execution report produced by --report"),
    ],
    top_n: Annotated[
        int,
        typer.Option(
            "--top",
            help="Number of top providers/models to display in breakdowns",
            min=1,
        ),
    ] = 10,
    providers_pricing: Annotated[
        str | None,
        typer.Option(
            "--providers-pricing",
            help="Path to providers.yml (pricing_per_million_tokens). "
            "Default search: ASK_LLM_PROVIDERS_YML, package root, ~/.config/ask_llm/providers.yml",
        ),
    ] = None,
) -> None:
    """Summarize an execution report and highlight failure patterns.

    Examples:
        ask-llm diagnose report.json
        ask-llm diagnose report.json --top 5
    """
    with cli_errors("diagnose"):
        path = Path(report_path).expanduser().resolve()
        if not path.exists():
            console.print_error(f"Report not found: {path}")
            raise typer.Exit(1)

        report = ExecutionReport.from_json_file(str(path))

        console.print(f"[bold]Execution Report[/bold]: {path}")
        console.print(f"Version: {report.version}")
        console.print(f"Command: {report.command}")
        if report.started_at:
            console.print(f"Started: {report.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if report.completed_at:
            duration = (report.completed_at - report.started_at).total_seconds()
            console.print(
                f"Completed: {report.completed_at.strftime('%Y-%m-%d %H:%M:%S')} ({duration:.1f}s)"
            )

        console.print()
        console.print("[bold]Summary[/bold]")
        console.print(f"  Total tasks: {report.total_tasks}")
        console.print(f"  Successful: {report.successful_tasks}")
        console.print(f"  Failed: {report.failed_tasks}")
        if report.total_tasks > 0:
            success_rate = report.successful_tasks / report.total_tasks * 100
            console.print(f"  Success rate: {success_rate:.1f}%")

        console.print()
        console.print("[bold]Token Usage[/bold]")
        console.print(f"  Input tokens: {report.token_summary.total_input_tokens:,}")
        console.print(f"  Output tokens: {report.token_summary.total_output_tokens:,}")
        total_tokens = (
            report.token_summary.total_input_tokens + report.token_summary.total_output_tokens
        )
        console.print(f"  Total tokens: {total_tokens:,}")

        # Provider/model breakdown
        model_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "success": 0, "failed": 0, "tokens": 0, "latency_ms": 0}
        )
        # E6/2.25: per-model token split so a cost estimate can be attached.
        model_token_split: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )
        for task in report.tasks:
            for attempt in task.attempts:
                key = f"{attempt.provider}/{attempt.model}"
                model_stats[key]["attempts"] += 1
                if attempt.status == TaskStatus.SUCCESS:
                    model_stats[key]["success"] += 1
                    model_stats[key]["tokens"] += (attempt.input_tokens or 0) + (
                        attempt.output_tokens or 0
                    )
                    model_stats[key]["latency_ms"] += int((attempt.latency or 0) * 1000)
                    model_token_split[key]["input"] += attempt.input_tokens or 0
                    model_token_split[key]["output"] += attempt.output_tokens or 0
                else:
                    model_stats[key]["failed"] += 1

        # E6/2.25: cost estimate from the pricing catalog, per provider/model.
        pricing_map, pricing_source = load_pricing_with_hint(providers_pricing)
        if pricing_map and model_token_split:
            console.print()
            console.print("[bold]Cost Estimate[/bold]")
            if pricing_source:
                console.print(f"  Pricing source: {pricing_source.name}")
            total_cost = 0.0
            any_price = False
            for key, split in sorted(model_token_split.items()):
                provider_name, model_name = key.split("/", 1)
                row = lookup_pricing(pricing_map, provider_name, model_name)
                if row is None:
                    continue
                any_price = True
                cost = estimate_cost_cny(row, split["input"], split["output"])
                total_cost += cost
                console.print(f"  {key}: ¥{cost:.4f}")
            if any_price:
                console.print(f"  [bold]Total: ¥{total_cost:.4f}[/bold]")
            else:
                console.print("  Unavailable — no pricing entries for the models used")

        if model_stats:
            console.print()
            console.print("[bold]Provider / Model Breakdown[/bold]")
            rows = []
            for key, stats in sorted(
                model_stats.items(), key=lambda x: x[1]["attempts"], reverse=True
            )[:top_n]:
                avg_latency = "-"
                if stats["success"] > 0:
                    avg_latency = f"{stats['latency_ms'] / stats['success']:.0f}ms"
                rows.append(
                    [
                        key,
                        stats["attempts"],
                        stats["success"],
                        stats["failed"],
                        f"{stats['tokens']:,}",
                        avg_latency,
                    ]
                )
            console.print_table(
                headers=[
                    "Provider/Model",
                    "Attempts",
                    "Success",
                    "Failed",
                    "Tokens",
                    "Avg Latency",
                ],
                rows=rows,
            )

        # Failure category breakdown
        if report.failure_summary.total_failed_tasks > 0:
            console.print()
            console.print("[bold]Failure Breakdown[/bold]")
            rows = []
            for category, count in sorted(
                report.failure_summary.by_category.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                rows.append([category, count])
            console.print_table(
                headers=["Category", "Tasks"],
                rows=rows,
            )

            # Terminal-failure hint
            terminal_count = sum(
                count
                for category, count in report.failure_summary.by_category.items()
                if category
                in {
                    ErrorCategory.AUTHENTICATION.value,
                    ErrorCategory.CONTENT_FILTER.value,
                    ErrorCategory.VALIDATION_ERROR.value,
                }
            )
            if terminal_count > 0:
                console.print()
                console.print_warning(
                    f"{terminal_count} task(s) failed with terminal error categories "
                    "(authentication, content-filter, validation). Fallback did not help these."
                )

        # Failed task details
        failed_tasks = [t for t in report.tasks if t.final_status == TaskStatus.FAILED]
        if failed_tasks:
            console.print()
            console.print(f"[bold]Failed Tasks ({len(failed_tasks)})[/bold]")
            rows = []
            for task in failed_tasks[:top_n]:
                category = task.final_error_category.value if task.final_error_category else "-"
                rows.append(
                    [
                        task.task_id,
                        f"{task.primary_provider}/{task.primary_model}",
                        category,
                        (task.final_error or "")[:80],
                    ]
                )
            console.print_table(
                headers=["Task ID", "Primary Model", "Category", "Error"],
                rows=rows,
            )
            if len(failed_tasks) > top_n:
                console.print_info(
                    f"... and {len(failed_tasks) - top_n} more failed tasks (use --top to show more)"
                )
