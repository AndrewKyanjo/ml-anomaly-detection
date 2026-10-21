"""Check training-run selection without network requests or registry access."""

import json
from pathlib import Path
import re
import subprocess
import textwrap

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docker-publish.yml"


@pytest.fixture
def training_run():
    return {
        "id": 24680,
        "status": "completed",
        "conclusion": "success",
        "head_sha": "abc123",
        "path": ".github/workflows/train.yml",
        "event": "workflow_dispatch",
        "head_repository": {"full_name": "owner/project"},
    }


@pytest.fixture
def execute_training_step(tmp_path, monkeypatch):
    match = re.search(
        r"^          python - <<'PY'\n(.*?)^          PY$",
        WORKFLOW.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "The publishing workflow must retain its Python run resolver."
    program = compile(textwrap.dedent(match.group(1)), str(WORKFLOW), "exec")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/project")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github-output"))

    def execute(responses, *, run_id=""):
        monkeypatch.setenv("RUN_ID", run_id)
        pending = iter(responses)
        calls = []

        def run(command, *, capture_output, text):
            assert capture_output and text
            calls.append(command)
            return next(pending)

        monkeypatch.setattr(subprocess, "run", run)
        exec(program, {"__name__": "__main__"})
        return calls

    return execute


def api_response(data=None, *, returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, json.dumps(data), stderr)


def test_automatically_selects_successful_training_for_current_commit(
    execute_training_step, training_run, tmp_path, capsys,
):
    calls = execute_training_step([
        api_response({"workflow_runs": [{"id": 24680}]}),
        api_response(training_run),
    ])

    assert calls == [
        [
            "gh", "api", "repos/owner/project/actions/workflows/train.yml/runs"
            "?head_sha=abc123&status=success&event=workflow_dispatch&per_page=1",
        ],
        ["gh", "api", "repos/owner/project/actions/runs/24680"],
    ]
    assert (tmp_path / "github-output").read_text() == "run_id=24680\n"
    assert "https://github.com/owner/project/actions/runs/24680" in capsys.readouterr().out


def test_explicit_run_id_is_validated_without_listing_runs(
    execute_training_step, training_run, tmp_path,
):
    calls = execute_training_step([api_response(training_run)], run_id=" 24680 ")

    assert calls == [["gh", "api", "repos/owner/project/actions/runs/24680"]]
    assert (tmp_path / "github-output").read_text() == "run_id=24680\n"


def test_missing_training_explains_next_step_without_setting_output(
    execute_training_step, tmp_path,
):
    with pytest.raises(SystemExit, match="Run 'Train and evaluate models'"):
        execute_training_step([api_response({"workflow_runs": []})])

    assert not (tmp_path / "github-output").exists()


@pytest.mark.parametrize("run_id", ["", "24680"])
def test_inaccessible_runs_fail_before_parsing_error_response(
    execute_training_step, run_id, tmp_path, capsys,
):
    error = api_response(
        {"message": "Not Found"}, returncode=1, stderr="gh: Not Found (HTTP 404)",
    )
    with pytest.raises(SystemExit, match="actual run ID from this repository"):
        execute_training_step([error], run_id=run_id)

    assert "HTTP 404" in capsys.readouterr().err
    assert not (tmp_path / "github-output").exists()


@pytest.mark.parametrize("invalid_fields", [
    {"head_sha": "different-commit"},
    {"path": ".github/workflows/ci.yml"},
    {"head_repository": {"full_name": "another/project"}},
    {"head_repository": None},
    {"event": "pull_request"},
    {"status": "in_progress"},
    {"conclusion": "failure"},
])
def test_rejects_untrusted_or_unsuccessful_training_without_setting_output(
    execute_training_step, training_run, invalid_fields, tmp_path,
):
    training_run.update(invalid_fields)
    with pytest.raises(SystemExit, match="exact commit being published"):
        execute_training_step([api_response(training_run)], run_id="24680")

    assert not (tmp_path / "github-output").exists()
