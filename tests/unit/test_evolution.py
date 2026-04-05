"""Tests for evolution layer: experience extraction, storage, retrieval, and self-evolution."""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from common.vector import get_vector_dim, ensure_collection, get_qdrant_client, text_to_embedding
from evolution.experience import extract_experience, search_similar_experiences, store_experience


# --- text_to_embedding tests ---

def test_embedding_dimension():
    vec = text_to_embedding("hello world")
    assert len(vec) == get_vector_dim()


def test_embedding_deterministic():
    v1 = text_to_embedding("test input")
    v2 = text_to_embedding("test input")
    assert v1 == v2


def test_embedding_different_texts():
    v1 = text_to_embedding("hello")
    v2 = text_to_embedding("goodbye")
    assert v1 != v2


def test_embedding_normalized():
    vec = text_to_embedding("normalize me")
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


# --- Qdrant in-memory tests ---

def test_qdrant_in_memory():
    client, _ = get_qdrant_client(in_memory=True)
    collections = client.get_collections().collections
    assert isinstance(collections, list)


def test_ensure_collection_creates():
    client, _ = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_collection")
    names = [c.name for c in client.get_collections().collections]
    assert "test_collection" in names


def test_ensure_collection_idempotent():
    client, _ = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_col")
    ensure_collection(client, "test_col")  # should not raise
    names = [c.name for c in client.get_collections().collections]
    assert names.count("test_col") == 1


# --- store + search round-trip ---

@pytest.mark.asyncio
async def test_store_and_search_experience():
    client, _ = get_qdrant_client(in_memory=True)
    collection = "test_experiences"
    ensure_collection(client, collection)

    experience = {
        "problem_pattern": "sorting algorithm implementation",
        "recommended_strategy": "single agent, direct code generation",
        "recommended_agents": ["code_agent"],
        "tips": "use built-in sorted() for simple cases",
    }
    embedding = text_to_embedding("write a sorting function in Python")

    point_id = await store_experience(client, collection, experience, embedding, "task-001")
    assert point_id  # non-empty string

    # Search with the same embedding → should find the experience
    results = await search_similar_experiences(client, collection, embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["problem_pattern"] == "sorting algorithm implementation"
    assert results[0]["score"] > 0.9  # near-exact match


@pytest.mark.asyncio
async def test_search_empty_collection():
    client, _ = get_qdrant_client(in_memory=True)
    collection = "empty_col"
    ensure_collection(client, collection)

    results = await search_similar_experiences(
        client, collection, text_to_embedding("anything"), top_k=3,
    )
    assert results == []


# --- extract_experience tests ---

@pytest.mark.asyncio
async def test_extract_experience_no_llm():
    result = await extract_experience({"user_input": "test"}, llm=None)
    assert result is None


@pytest.mark.asyncio
async def test_extract_experience_with_mock_llm():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content='{"problem_pattern": "test", "recommended_strategy": "direct", "recommended_agents": ["code_agent"], "tips": "none"}'
    )

    result = await extract_experience(
        {"user_input": "write code", "intent": "code", "subtasks": [], "status": "completed"},
        llm=mock_llm,
    )
    assert result is not None
    assert result["problem_pattern"] == "test"
    assert result["recommended_agents"] == ["code_agent"]


# ══════════════════════════════════════════════════════════════════
# Self-Evolution System Tests
# ══════════════════════════════════════════════════════════════════

from evolution.diagnostician import Diagnostician, DiagnosticReport
from evolution.prescriber import Prescriber, Evolution
from evolution.applier import Applier, ApplyResult


# ── Diagnostician ────────────────────────────────────────────────

def test_diagnostic_report_summary():
    report = DiagnosticReport(
        total_tasks=100, success_rate=0.85, avg_duration_s=45.2,
        timeout_rate=0.05,
        weak_areas=[{"category": "coding", "failure_rate": 0.3, "count": 10}],
        diagnosed_at="2026-03-26T10:00:00",
    )
    summary = report.summary()
    assert "100" in summary
    assert "85%" in summary
    assert "coding" in summary


def test_diagnostic_report_to_dict():
    report = DiagnosticReport(total_tasks=50, success_rate=0.9)
    d = report.to_dict()
    assert d["total_tasks"] == 50
    assert d["success_rate"] == 0.9


def test_diagnostician_categorize():
    diag = Diagnostician.__new__(Diagnostician)
    assert diag._categorize_input("帮我写一段代码") == "coding"
    assert diag._categorize_input("搜索最新天气") == "search"
    assert diag._categorize_input("写一篇文章") == "writing"
    assert diag._categorize_input("分析这个数据") == "analysis"
    assert diag._categorize_input("你好") == "general"


@pytest.mark.asyncio
async def test_diagnostician_empty():
    """Diagnostician should produce valid report even with no data."""
    with tempfile.TemporaryDirectory() as tmp:
        diag = Diagnostician(data_dir=Path(tmp))
        # Mock external data sources to isolate test
        diag._analyze_traces = AsyncMock(return_value={
            "total_tasks": 0, "success_rate": 0, "avg_duration_s": 0,
            "timeout_rate": 0, "weak_areas": [], "failing_patterns": [],
        })
        diag._analyze_conversations = MagicMock(return_value={
            "total_tasks": 0, "success_rate": 0, "weak_areas": [],
            "missing_capabilities": [], "repeated_questions": [],
        })
        report = await diag.diagnose(days=7)
        assert isinstance(report, DiagnosticReport)
        assert report.total_tasks == 0


# ── Prescriber ───────────────────────────────────────────────────

def test_prescribe_timeout_tuning():
    prescriber = Prescriber()
    evolutions = prescriber._prescribe_timeout_tuning({"timeout_rate": 0.25})
    assert len(evolutions) == 1
    assert evolutions[0].type == "tune_params"
    assert evolutions[0].priority == 1


def test_prescribe_no_tuning_ok():
    prescriber = Prescriber()
    evolutions = prescriber._prescribe_timeout_tuning({"timeout_rate": 0.05})
    assert len(evolutions) == 0


def test_prescribe_underused_tools():
    prescriber = Prescriber()
    evolutions = prescriber._prescribe_underused_tools(
        {"underused_tools": [f"tool_{i}" for i in range(8)]}
    )
    assert len(evolutions) == 1
    assert evolutions[0].type == "update_knowledge"


@pytest.mark.asyncio
async def test_prescribe_without_llm():
    prescriber = Prescriber()
    evolutions = await prescriber.prescribe(
        {"timeout_rate": 0.3, "underused_tools": [f"t{i}" for i in range(10)]},
        llm=None,
    )
    assert len(evolutions) >= 1


def test_evolution_to_dict():
    evo = Evolution(type="add_skill", target="t", reason="r",
                    patch={"name": "x"}, confidence=0.8)
    d = evo.to_dict()
    assert d["type"] == "add_skill"
    assert d["confidence"] == 0.8


def test_evolution_summary():
    evo = Evolution(type="tune_params", target="react.timeout",
                    reason="High timeout rate", patch={}, confidence=0.9)
    assert "tune_params" in evo.summary()
    assert "90%" in evo.summary()


# ── Applier ──────────────────────────────────────────────────────

def _make_applier(tmp):
    root = Path(tmp)
    (root / "config" / "skills").mkdir(parents=True)
    (root / "config" / "knowledge").mkdir(parents=True)
    (root / "data" / "evolution_backups").mkdir(parents=True)
    return Applier(project_root=root), root


def test_applier_add_skill_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        applier, root = _make_applier(tmp)
        evo = Evolution(type="add_skill", target="config/skills/test_skill",
                        reason="test", patch={"name": "test", "description": "d", "prompt": "p"})
        result = applier._apply_one(evo, dry_run=True)
        assert result.success
        assert "dry-run" in result.message
        assert not (root / "config" / "skills" / "test_skill").exists()


def test_applier_add_skill():
    with tempfile.TemporaryDirectory() as tmp:
        applier, root = _make_applier(tmp)
        evo = Evolution(type="add_skill", target="config/skills/my_skill",
                        reason="test", patch={"name": "MySkill", "description": "d", "prompt": "hello"})
        result = applier._apply_one(evo, dry_run=False)
        assert result.success
        skill_md = root / "config" / "skills" / "my_skill" / "SKILL.md"
        assert skill_md.exists()
        assert "MySkill" in skill_md.read_text()


def test_applier_add_tool():
    with tempfile.TemporaryDirectory() as tmp:
        applier, root = _make_applier(tmp)
        evo = Evolution(type="add_tool", target="config/tools.yaml", reason="test",
                        patch={"name": "weather", "endpoint": "http://x.com", "method": "GET"})
        result = applier._apply_one(evo, dry_run=False)
        assert result.success
        import yaml
        data = yaml.safe_load((root / "config" / "tools.yaml").read_text())
        assert any(t["name"] == "weather" for t in data["tools"])


def test_applier_update_knowledge():
    with tempfile.TemporaryDirectory() as tmp:
        applier, root = _make_applier(tmp)
        evo = Evolution(type="update_knowledge", target="config/knowledge/test.md",
                        reason="test", patch="# Test\nContent.")
        result = applier._apply_one(evo, dry_run=False)
        assert result.success
        assert (root / "config" / "knowledge" / "test.md").exists()


def test_applier_duplicate_skill():
    with tempfile.TemporaryDirectory() as tmp:
        applier, root = _make_applier(tmp)
        (root / "config" / "skills" / "existing").mkdir()
        evo = Evolution(type="add_skill", target="config/skills/existing",
                        reason="test", patch="")
        result = applier._apply_one(evo, dry_run=False)
        assert not result.success
        assert "already exists" in result.message


def test_applier_unknown_type():
    with tempfile.TemporaryDirectory() as tmp:
        applier, _ = _make_applier(tmp)
        evo = Evolution(type="modify_source", target="", reason="", patch="")
        result = applier._apply_one(evo, dry_run=False)
        assert not result.success


# ── Controller ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_controller_propose():
    with patch("evolution.controller.EvolutionController.__init__", return_value=None):
        ctrl = _mock_controller([
            Evolution(type="tune_params", target="t", reason="r", patch={}, confidence=0.9),
        ])
        report = await ctrl.evolve(mode="propose")
        assert report.mode == "propose"
        assert len(report.proposed) == 1
        assert len(report.applied) == 0


@pytest.mark.asyncio
async def test_controller_auto():
    with patch("evolution.controller.EvolutionController.__init__", return_value=None):
        ctrl = _mock_controller([
            Evolution(type="tune_params", target="t", reason="safe", patch={}, confidence=0.9),
            Evolution(type="add_skill", target="s", reason="confirm", patch={}, confidence=0.8),
        ])
        ctrl.applier.apply = AsyncMock(return_value=[
            ApplyResult("tune_params", "t", True, "ok"),
        ])
        report = await ctrl.evolve(mode="auto")
        assert len(report.applied) == 1
        assert len(report.skipped) == 1


@pytest.mark.asyncio
async def test_controller_confidence_filter():
    with patch("evolution.controller.EvolutionController.__init__", return_value=None):
        ctrl = _mock_controller([
            Evolution(type="tune_params", target="t", reason="high", patch={}, confidence=0.9),
            Evolution(type="tune_params", target="t2", reason="low", patch={}, confidence=0.2),
        ])
        report = await ctrl.evolve(mode="propose", min_confidence=0.5)
        assert len(report.proposed) == 1


def _mock_controller(evolutions):
    from evolution.controller import EvolutionController
    ctrl = EvolutionController.__new__(EvolutionController)
    ctrl.diagnostician = MagicMock()
    ctrl.prescriber = MagicMock()
    ctrl.applier = MagicMock()
    ctrl.absorber = MagicMock()
    ctrl.diagnostician.diagnose = AsyncMock(
        return_value=DiagnosticReport(total_tasks=10, success_rate=0.8)
    )
    ctrl.prescriber.prescribe = AsyncMock(return_value=evolutions)
    ctrl._save_evolution_log = MagicMock()
    return ctrl


# ── Absorber ─────────────────────────────────────────────────────

def test_absorber_analyze_structure():
    from evolution.absorber import Absorber
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "src").mkdir()
        (repo / "src" / "main.py").write_text("from fastapi import FastAPI")
        (repo / "requirements.txt").write_text("fastapi\nuvicorn\n")
        (repo / "README.md").write_text("# My Project\nA cool API.")
        (repo / "Dockerfile").write_text("FROM python:3.12")

        absorber = Absorber()
        analysis = absorber._analyze_structure(repo)
        assert analysis["has_readme"]
        assert analysis["has_api"]
        assert analysis["has_dockerfile"]
        assert "python" in analysis["languages"]


def test_absorber_rule_based():
    from evolution.absorber import Absorber
    absorber = Absorber()
    analysis = {"name": "test", "has_api": True, "has_cli": True,
                "has_pyproject": True, "has_dockerfile": False,
                "languages": ["python"], "readme_content": "docs", "files": []}
    caps = absorber._rule_based_analyze(analysis)
    types = [c["integration_type"] for c in caps]
    assert "api_tool" in types
    assert "cli_skill" in types


def test_absorber_generate_evolutions():
    from evolution.absorber import Absorber
    absorber = Absorber()
    caps = [
        {"name": "myapi", "description": "API", "integration_type": "api_tool",
         "confidence": 0.8, "api_endpoint": "http://localhost:3000"},
        {"name": "mydocs", "description": "Docs", "integration_type": "knowledge",
         "confidence": 0.7, "doc_content": "# Hello"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        evos = absorber._generate_evolutions(caps, {}, Path(tmp))
        assert len(evos) == 2
        assert evos[0].type == "add_tool"
        assert evos[1].type == "update_knowledge"
