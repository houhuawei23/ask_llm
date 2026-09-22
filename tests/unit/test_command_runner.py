"""Unit tests for the shared checkpoint lifecycle (run_with_checkpoint, H9)."""

from pathlib import Path

import pytest

from ask_llm.config.manager import ConfigManager
from ask_llm.core.batch_checkpoint import BatchCheckpoint
from ask_llm.core.batch_models import BatchResult, BatchTask, ModelConfig, TaskStatus
from ask_llm.core.command_runner import compute_checkpoint_digest, run_with_checkpoint


def _task(i: int) -> BatchTask:
    return BatchTask(
        task_id=i,
        prompt="Translate: {content}",
        content=f"chunk {i}",
        model_settings=ModelConfig(provider="test", model="test-model"),
    )


def _success(i: int) -> BatchResult:
    from ask_llm.core.models import RequestMetadata

    return BatchResult(
        task_id=i,
        prompt="p",
        content="c",
        model_settings=ModelConfig(provider="test", model="test-model"),
        response="ok",
        metadata=RequestMetadata(
            provider="test",
            model="test-model",
            temperature=0.7,
            input_tokens=1,
            output_tokens=1,
            latency=0.1,
        ),
        status=TaskStatus.SUCCESS,
    )


@pytest.fixture
def config_manager(app_config):
    return ConfigManager(app_config)


class TestComputeCheckpointDigest:
    def test_same_content_same_digest_different_path(self, tmp_path):
        a = tmp_path / "a.yml"
        b = tmp_path / "b.yml"
        a.write_text("k: v", encoding="utf-8")
        b.write_text("k: v", encoding="utf-8")
        assert compute_checkpoint_digest(a) == compute_checkpoint_digest(b)

    def test_edited_content_changes_digest(self, tmp_path):
        f = tmp_path / "in.md"
        f.write_text("v1", encoding="utf-8")
        d1 = compute_checkpoint_digest(f)
        f.write_text("v2", encoding="utf-8")
        assert compute_checkpoint_digest(f) != d1

    def test_task_payload_part_of_digest(self):
        t1, t2 = _task(0), _task(0)
        assert compute_checkpoint_digest(None, [t1]) == compute_checkpoint_digest(None, [t2])
        t2.content = "edited"
        assert compute_checkpoint_digest(None, [t1]) != compute_checkpoint_digest(None, [t2])

    def test_model_change_changes_digest(self):
        t1 = _task(0)
        t2 = _task(0)
        t2.model_settings = ModelConfig(provider="test", model="other-model")
        assert compute_checkpoint_digest(None, [t1]) != compute_checkpoint_digest(None, [t2])


class TestResumeValidation:
    def _make_checkpoint(self, checkpoint_path: Path, digest: str, completed: list[int]) -> None:
        cp = BatchCheckpoint.create(command="batch", config_digest=digest)
        cp.completed_task_ids = completed
        cp.successful_results = [_success(i) for i in completed]
        cp.save(checkpoint_path)

    def test_matching_digest_resumes_and_filters_completed(
        self, tmp_path, config_manager, monkeypatch
    ):
        input_file = tmp_path / "config.yml"
        input_file.write_text("tasks: 2", encoding="utf-8")
        digest = compute_checkpoint_digest(input_file)
        checkpoint_path = tmp_path / "cp.json"
        self._make_checkpoint(checkpoint_path, digest, completed=[0])

        captured = {}

        def fake_runner(tasks, *args, **kwargs):
            captured["tasks"] = tasks
            return [_success(t.task_id) for t in tasks], None

        monkeypatch.setattr("ask_llm.core.command_runner.run_global_batch_tasks", fake_runner)
        outcome = run_with_checkpoint(
            command="batch",
            config_digest=digest,
            checkpoint_path=str(checkpoint_path),
            tasks=[_task(0), _task(1)],
            config_manager=config_manager,
            resume=True,
            max_retries=1,
            max_workers=2,
        )
        assert [t.task_id for t in captured["tasks"]] == [1]
        assert not outcome.all_previously_completed
        assert outcome.checkpoint_deleted  # clean full success

    def test_digest_mismatch_refuses_resume(self, tmp_path, config_manager, monkeypatch):
        input_file = tmp_path / "config.yml"
        input_file.write_text("tasks: 2", encoding="utf-8")
        checkpoint_path = tmp_path / "cp.json"
        self._make_checkpoint(checkpoint_path, compute_checkpoint_digest(input_file), completed=[0])
        # Input edited after the checkpoint was written.
        input_file.write_text("tasks: 2 EDITED", encoding="utf-8")

        def boom(*args, **kwargs):
            raise AssertionError("runner must not be called on refused resume")

        monkeypatch.setattr("ask_llm.core.command_runner.run_global_batch_tasks", boom)
        with pytest.raises(ValueError, match="does not match"):
            run_with_checkpoint(
                command="batch",
                config_digest=compute_checkpoint_digest(input_file),
                checkpoint_path=str(checkpoint_path),
                tasks=[_task(0), _task(1)],
                config_manager=config_manager,
                resume=True,
                max_retries=1,
                max_workers=2,
            )

    def test_legacy_path_string_checkpoint_refuses_resume(
        self, tmp_path, config_manager, monkeypatch
    ):
        """Pre-H9 checkpoints stored the config *path* as the digest; they must
        be rejected (re-run fresh) instead of mis-mapping old results."""
        checkpoint_path = tmp_path / "cp.json"
        self._make_checkpoint(checkpoint_path, str(tmp_path / "config.yml"), completed=[0])

        def boom(*args, **kwargs):
            raise AssertionError("runner must not be called on refused resume")

        monkeypatch.setattr("ask_llm.core.command_runner.run_global_batch_tasks", boom)
        with pytest.raises(ValueError, match="does not match"):
            run_with_checkpoint(
                command="batch",
                config_digest=compute_checkpoint_digest(None, [_task(0), _task(1)]),
                checkpoint_path=str(checkpoint_path),
                tasks=[_task(0), _task(1)],
                config_manager=config_manager,
                resume=True,
                max_retries=1,
                max_workers=2,
            )

    def test_command_mismatch_refuses_resume(self, tmp_path, config_manager, monkeypatch):
        checkpoint_path = tmp_path / "cp.json"
        digest = compute_checkpoint_digest(None, [_task(0)])
        self._make_checkpoint(checkpoint_path, digest, completed=[0])
        # Checkpoint written for "batch", resume attempted for "trans".

        def boom(*args, **kwargs):
            raise AssertionError("runner must not be called on refused resume")

        monkeypatch.setattr("ask_llm.core.command_runner.run_global_batch_tasks", boom)
        with pytest.raises(ValueError, match="does not match"):
            run_with_checkpoint(
                command="trans",
                config_digest=digest,
                checkpoint_path=str(checkpoint_path),
                tasks=[_task(0)],
                config_manager=config_manager,
                resume=True,
                max_retries=1,
                max_workers=2,
            )


class TestNonResumeCheckpointBackup:
    def test_non_resume_rerun_preserves_prior_checkpoint_as_bak(
        self, tmp_path, config_manager, monkeypatch
    ):
        """2.4: a fresh (non-resume) rerun keeps the prior checkpoint as .bak."""
        checkpoint_path = tmp_path / "cp.json"
        digest = "digest"
        self._make_checkpoint(checkpoint_path, digest, completed=[0])

        monkeypatch.setattr(
            "ask_llm.core.command_runner.run_global_batch_tasks",
            lambda tasks, *a, **k: ([_success(t.task_id) for t in tasks], None),
        )
        run_with_checkpoint(
            command="batch",
            config_digest=digest,
            checkpoint_path=str(checkpoint_path),
            tasks=[_task(0)],
            config_manager=config_manager,
            resume=False,  # fresh run over the existing checkpoint
            max_retries=1,
            max_workers=1,
        )
        # New checkpoint written for this run; prior one preserved as .bak.
        assert Path(str(checkpoint_path) + ".bak").exists()
        bak = BatchCheckpoint.load(str(checkpoint_path) + ".bak")
        assert 0 in bak.completed_task_ids

    def _make_checkpoint(self, path, digest, completed):
        cp = BatchCheckpoint.create(command="batch", config_digest=digest)
        cp.merge([_success(i) for i in completed])
        cp.save(path)
