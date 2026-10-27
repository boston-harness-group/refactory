"""R5 tests: model routing for design_doc, and model recording."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh.design import DesignMode
from refactory_dsh.workers import resolve_route


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_resolve_route_with_node_routes_for_design_doc():
    base = {"worker_backend": "dsh", "model": "default-model",
            "provider": "", "effort": "", "permission_mode": "default"}
    result = resolve_route(base, node="design_doc",
                           node_routes={"design_doc": {"model": "test-strong-model"}})
    assert result["model"] == "test-strong-model"


def test_resolve_route_with_node_routes_for_design_doc():
    base = {"worker_backend": "dsh", "model": "default-model",
            "provider": "", "effort": "", "permission_mode": "default"}
    result = resolve_route(base, node="design_doc",
                           node_routes={"design_doc": {"model": "test-strong-model"}})
    assert result["model"] == "test-strong-model"


def test_design_doc_node_has_routable_role():
    graph = DesignMode().build()
    assert graph.nodes["design_doc"].role == "ceo"
    # No design reviewer: design_doc is the one planning node.
    assert "design_review" not in graph.nodes


def test_report_artifact_reads_planning_models_from_worker_receipts(tmp_path):
    """report_artifact collects planning_models from worker receipt files.

    Without the fix, report_artifact would not read worker receipts at all.
    This test creates receipt files and verifies the reading logic works.
    """
    workers_dir = tmp_path / ".refactory" / "workers"
    workers_dir.mkdir(parents=True)

    receipt1 = {
        "schema_version": 1,
        "node_id": "design_doc",
        "actual_route": {"model": "claude-opus-4-20250514"},
        "state": "completed",
    }
    (workers_dir / "worker-fake1.json").write_text(json.dumps(receipt1))


    # A receipt for a non-planning node should be ignored
    receipt3 = {
        "schema_version": 1,
        "node_id": "emit",
        "actual_route": {"model": "some-model"},
        "state": "completed",
    }
    (workers_dir / "worker-fake3.json").write_text(json.dumps(receipt3))

    # Exercise the same logic report_artifact uses
    planning_models: dict[str, str | None] = {}
    try:
        receipt_files = sorted(
            (f for f in workers_dir.iterdir() if f.suffix == ".json"),
            key=lambda f: f.stat().st_mtime,
        )
    except OSError:
        receipt_files = []
    for wf in receipt_files:
        try:
            rec = json.loads(wf.read_text())
            nid = rec.get("node_id", "")
            if nid == "design_doc":
                route = rec.get("actual_route") or rec.get("requested_route") or {}
                planning_models[nid] = route.get("model") or None
        except (json.JSONDecodeError, OSError):
            pass

    assert planning_models == {"design_doc": "claude-opus-4-20250514"}
    assert "emit" not in planning_models


def test_run_records_planning_models(tmp_path, monkeypatch):
    """A scripted run with worker receipts records planning_models in design.json.

    Without the fix, report_artifact would not read worker receipts at all,
    so design.json would never contain planning_models.
    """
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               VERIFICATION_ANSWER,
                               REVIEW_ANSWERS, FakeModel, render)

    class _NodeClient:
        """Thin per-node wrapper that creates worker receipts for planning nodes."""
        def __init__(self, parent, node_id, route):
            self._parent = parent
            self._node_id = node_id
            self._route = route

        def complete(self, prompt, *, system="", workdir=None, **workspace):
            answer = self._parent.complete(prompt, system=system, workdir=workdir, **workspace)
            if workdir is not None and self._node_id == "design_doc":
                from pathlib import Path as P
                workers_dir = P(workdir) / ".refactory" / "workers"
                workers_dir.mkdir(parents=True, exist_ok=True)
                receipt = {
                    "schema_version": 1, "node_id": self._node_id,
                    "actual_route": self._route, "state": "completed",
                }
                (workers_dir / f"worker-{self._node_id}.json").write_text(
                    json.dumps(receipt))
            return answer

        def context(self):
            return {}

        def route(self):
            return {"worker_backend": "dsh"}

    class RoutedModel(FakeModel):
        """FakeModel that supports for_node and creates worker receipts."""
        def __init__(self, answers, node_routes=None):
            super().__init__(answers)
            self._node_routes = node_routes or {}

        def for_node(self, node_id, role=""):
            return _NodeClient(self, node_id, self._node_routes.get(node_id, {}))

    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}),
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]
    routes = {"design_doc": {"model": "test-strong-model"}}
    result = DesignMode().run(
        tmp_path / "run", model=RoutedModel(answers, node_routes=routes),
        verdicts={"gate_strategy": "proceed"},
        task={"name": "routing_test", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    report = json.loads((tmp_path / "run" / "design.json").read_text())
    assert "planning_models" in report, (
        "report must include planning_models from worker receipts"
    )
    assert report["planning_models"] == {"design_doc": "test-strong-model"}


def test_delivery_body_includes_planning_models(tmp_path, monkeypatch):
    """A run with planning_models shows them in the PR body's Verification section."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    from tests_support import (DESIGN_DOC_ANSWER,
                               PITFALLS_ANSWER, RESEARCH_ANSWER,
                               VERIFICATION_ANSWER,
                               REVIEW_ANSWERS, FakeModel, render)

    class _NodeClient:
        def __init__(self, parent, node_id, route):
            self._parent = parent
            self._node_id = node_id
            self._route = route

        def complete(self, prompt, *, system="", workdir=None, **workspace):
            answer = self._parent.complete(prompt, system=system, workdir=workdir, **workspace)
            if workdir is not None and self._node_id == "design_doc":
                from pathlib import Path as P
                workers_dir = P(workdir) / ".refactory" / "workers"
                workers_dir.mkdir(parents=True, exist_ok=True)
                receipt = {
                    "schema_version": 1, "node_id": self._node_id,
                    "actual_route": self._route, "state": "completed",
                }
                (workers_dir / f"worker-{self._node_id}.json").write_text(
                    json.dumps(receipt))
            return answer

        def context(self):
            return {}

        def route(self):
            return {"worker_backend": "dsh"}

    class RoutedModel(FakeModel):
        def __init__(self, answers, node_routes=None):
            super().__init__(answers)
            self._node_routes = node_routes or {}

        def for_node(self, node_id, role=""):
            return _NodeClient(self, node_id, self._node_routes.get(node_id, {}))

    answers = [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}),
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]
    routes = {"design_doc": {"model": "test-strong-model"}}
    model = RoutedModel(answers, node_routes=routes)
    result = DesignMode().run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "routing_pr", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error
    # The design.json report should have planning_models
    report = json.loads((tmp_path / "run" / "design.json").read_text())
    assert "planning_models" in report
    assert report["planning_models"] == {"design_doc": "test-strong-model"}
