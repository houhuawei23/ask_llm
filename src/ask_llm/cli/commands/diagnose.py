"""Typer command `diagnose` — inspect execution reports produced by batch/trans/paper."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from ask_llm.cli.errors import raise_unexpected_cli_error
from ask_llm.core.execution_report import ExecutionReport
from ask_llm.core.telemetry import ErrorCategory
from ask_llm.utils.console import console


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
) -> None:
    """Summarize an execution report and highlight failure patterns.

    Examples:
        ask-llm diagnose report.json
        ask-llm diagnose report.json --top 5
    """
    try:
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
        for task in report.tasks:
            for attempt in task.attempts:
                key = f"{attempt.provider}/{attempt.model}"
                model_stats[key]["attempts"] += 1
                if attempt.status.value == "success":
                    model_stats[key]["success"] += 1
                    model_stats[key]["tokens"] += (attempt.input_tokens or 0) + (
                        attempt.output_tokens or 0
                    )
                    model_stats[key]["latency_ms"] += int((attempt.latency or 0) * 1000)
                else:
                    model_stats[key]["failed"] += 1

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
        failed_tasks = [t for t in report.tasks if t.final_status.value == "failed"]
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

    except FileNotFoundError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print_error(f"Invalid report: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        raise_unexpected_cli_error("diagnose", e)
