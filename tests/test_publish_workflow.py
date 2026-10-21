"""Check self-hosted workflow file staging without private data or Docker."""

from pathlib import Path
import re
import shutil
import textwrap
from unittest.mock import Mock

import pytest


WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"
TRAINING_FILES = (
    "data/raw/data.csv",
    "data/processed/split.parquet",
    "data/processed/validation_split.parquet",
)
OUTPUT_FILES = (
    "models/loan_models.joblib",
    "data/processed/split.parquet",
    "data/processed/validation_split.parquet",
    "reports/validation_metrics.csv",
    "reports/test_model_comparison.csv",
)


@pytest.fixture
def staging(tmp_path, monkeypatch):
    project = tmp_path / "local project"
    checkout = tmp_path / "runner checkout"
    project.mkdir()
    checkout.mkdir()
    monkeypatch.setenv("ML_PROJECT_PATH", str(project))
    monkeypatch.chdir(checkout)

    def execute(workflow="train.yml", step=0):
        path = WORKFLOWS / workflow
        programs = re.findall(
            r"^          @'\n(.*?)^          '@ \| python -$",
            path.read_text(encoding="utf-8"),
            re.MULTILINE | re.DOTALL,
        )
        assert len(programs) > step, "The workflow must contain its file staging step."
        program = compile(textwrap.dedent(programs[step]), str(path), "exec")
        exec(program, {"__name__": "__main__"})

    return execute, project, checkout


def write_file(root, name, content=b"fixture contents"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("workflow", ["train.yml", "docker-publish.yml"])
@pytest.mark.parametrize("configured", [None, "", " \t\n"])
def test_missing_project_configuration_fails_before_copy(
    staging, monkeypatch, workflow, configured,
):
    execute, _, checkout = staging
    if configured is None:
        monkeypatch.delenv("ML_PROJECT_PATH")
    else:
        monkeypatch.setenv("ML_PROJECT_PATH", configured)

    with pytest.raises(SystemExit, match="ML_PROJECT_PATH"):
        execute(workflow)

    assert list(checkout.iterdir()) == []


@pytest.mark.parametrize("workflow", ["train.yml", "docker-publish.yml"])
def test_missing_local_file_fails_with_useful_message(staging, workflow):
    execute, _, checkout = staging

    with pytest.raises(SystemExit, match="Local .*not found"):
        execute(workflow)

    assert list(checkout.iterdir()) == []


def test_training_copies_csv_and_persisted_splits_without_mutating_source(staging):
    execute, project, checkout = staging
    source_files = {
        name: write_file(project, name, f"contents of {name}".encode())
        for name in TRAINING_FILES
    }
    originals = {
        name: (path.read_bytes(), path.stat().st_mtime_ns)
        for name, path in source_files.items()
    }
    write_file(project, "private.txt", b"keep outside checkout")

    execute()

    for name, path in source_files.items():
        content, modified = originals[name]
        assert (checkout / name).read_bytes() == content
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == modified
    assert not (checkout / "private.txt").exists()


def test_training_succeeds_without_optional_saved_splits(staging, monkeypatch):
    execute, project, checkout = staging
    write_file(project, "data/raw/data.csv", b"feature,target\n1,0\n")
    monkeypatch.setenv("ML_PROJECT_PATH", f"  {project}  ")

    execute()

    assert (checkout / "data/raw/data.csv").read_bytes() == b"feature,target\n1,0\n"
    assert not (checkout / "data/processed").exists()


def test_publish_copies_local_model_without_mutating_it(staging):
    execute, project, checkout = staging
    source = write_file(project, "models/loan_models.joblib", b"trusted fitted artifact")
    modified = source.stat().st_mtime_ns

    execute("docker-publish.yml")

    assert (checkout / "models/loan_models.joblib").read_bytes() == b"trusted fitted artifact"
    assert source.read_bytes() == b"trusted fitted artifact"
    assert source.stat().st_mtime_ns == modified


@pytest.mark.parametrize("workflow,name", [
    ("train.yml", "data/raw/data.csv"),
    ("docker-publish.yml", "models/loan_models.joblib"),
])
def test_staging_same_file_skips_copy(staging, monkeypatch, workflow, name):
    execute, project, _ = staging
    source = write_file(project, name)
    monkeypatch.chdir(project)
    copy = Mock(side_effect=AssertionError("Same-file copying must be skipped."))
    monkeypatch.setattr(shutil, "copy2", copy)

    execute(workflow)

    copy.assert_not_called()
    assert source.read_bytes() == b"fixture contents"


def test_training_outputs_are_saved_back_to_local_project(staging):
    execute, project, checkout = staging
    for name in OUTPUT_FILES:
        write_file(checkout, name, f"new {name}".encode())
        write_file(project, name, b"previous artifact")

    execute(step=1)

    for name in OUTPUT_FILES:
        content = f"new {name}".encode()
        assert (project / name).read_bytes() == content
        assert (checkout / name).read_bytes() == content
    assert list(project.rglob("*.tmp")) == []


def test_failed_output_copy_preserves_previous_local_model(staging, monkeypatch):
    execute, project, checkout = staging
    write_file(checkout, OUTPUT_FILES[0], b"new model")
    original = write_file(project, OUTPUT_FILES[0], b"previous model")
    monkeypatch.setattr(shutil, "copy2", Mock(side_effect=OSError("Copy interrupted")))

    with pytest.raises(OSError, match="Copy interrupted"):
        execute(step=1)

    assert original.read_bytes() == b"previous model"
