"""Performance benchmarks for Phase G optimizations.

Run selectively with:
    pytest tests/benchmarks/test_performance.py --benchmark-only
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ask_llm.core.batch_models import BatchResult, ModelConfig, TaskStatus
from ask_llm.core.models import RequestMetadata
from ask_llm.utils.batch_exporter import BatchResultExporter
from ask_llm.utils.provider_cache import ProviderAdapterCache


def _make_results(count: int) -> list[BatchResult]:
    """Generate a list of BatchResults for benchmarking."""
    return [
        BatchResult(
            task_id=i,
            prompt="Translate the following text to Chinese.",
            content="Hello world" * 50,
            output_filename=None,
            model_settings=ModelConfig(provider="openai", model="gpt-4"),
            response="你好世界" * 50,
            status=TaskStatus.SUCCESS,
            metadata=RequestMetadata(
                provider="openai",
                model="gpt-4",
                temperature=0.7,
                input_tokens=100,
                output_tokens=100,
                latency=1.0,
                timestamp=datetime.now(),
            ),
        )
        for i in range(count)
    ]


def _make_exporter(count: int) -> BatchResultExporter:
    results = _make_results(count)
    statistics = MagicMock()
    statistics.total_tasks = count
    statistics.successful_tasks = count
    statistics.failed_tasks = 0
    statistics.average_latency = 1.0
    statistics.total_input_tokens = count * 100
    statistics.total_output_tokens = count * 100
    return BatchResultExporter(results, statistics)


@pytest.mark.benchmark
@pytest.mark.parametrize("count", [100, 1000])
def test_benchmark_json_export_streaming(benchmark, count):
    """Benchmark streaming JSON export for large result sets."""
    exporter = _make_exporter(count)

    def export():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
            exporter._export_json(f.name)

    benchmark(export)


@pytest.mark.benchmark
@pytest.mark.parametrize("count", [100, 1000])
def test_benchmark_json_export_dumps(benchmark, count):
    """Baseline: json.dumps export for large result sets."""
    exporter = _make_exporter(count)

    def export():
        data = exporter._prepare_data()
        json.dumps(data, indent=2, ensure_ascii=False, default=str)

    benchmark(export)


@pytest.mark.benchmark
def test_benchmark_provider_adapter_cache_hit(benchmark):
    """Benchmark provider adapter cache hit path."""
    ProviderAdapterCache.clear()
    config = {
        "api_provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "models": ["gpt-4"],
        "api_temperature": 0.7,
        "api_top_p": None,
        "max_tokens": None,
        "timeout": 60.0,
    }
    with patch("ask_llm.utils.provider_cache._create_cached_adapter") as mock_create:
        mock_create.return_value = MagicMock()
        ProviderAdapterCache.get(config, default_model="gpt-4")

    def access():
        ProviderAdapterCache.get(config, default_model="gpt-4")

    benchmark(access)
    ProviderAdapterCache.clear()
