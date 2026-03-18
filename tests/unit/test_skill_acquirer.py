"""Tests for skill acquirer — 3-level acquisition chain."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.skill_acquirer import SkillAcquirer, AcquisitionLevel, AcquisitionResult
from agents.skill_security import (
    RiskLevel, SecurityVerdict, quick_static_check, review_skill,
)


# === SkillAcquirer tests ===

@pytest.mark.asyncio
async def test_acquire_falls_through_all_levels():
    """With no backends, all levels should return not found."""
    acquirer = SkillAcquirer()
    with patch("shutil.which", return_value=None):  # disable mcporter
        result = await acquirer.acquire("build a website", ["web_dev"])
    assert result.found is False
    assert result.level == AcquisitionLevel.EXTERNAL_SKILL  # last level


@pytest.mark.asyncio
async def test_acquire_result_structure():
    acquirer = SkillAcquirer()
    result = await acquirer.acquire("test task", ["skill_a"])
    assert hasattr(result, "level")
    assert hasattr(result, "found")
    assert hasattr(result, "content")


@pytest.mark.asyncio
async def test_l1_history_with_qdrant():
    """L1 should return experience when Qdrant has matching data."""
    from common.vector import get_qdrant_client, ensure_collection, text_to_embedding
    from evolution.experience import store_experience

    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_skills")

    # Pre-store an experience
    exp = {
        "problem_pattern": "sorting algorithm",
        "recommended_strategy": "use code_agent with direct generation",
        "recommended_agents": ["code_agent"],
        "tips": "Python has built-in sorted()",
    }
    await store_experience(client, "test_skills", exp, text_to_embedding("write a sort function"), "t1")

    acquirer = SkillAcquirer(qdrant=client, collection_name="test_skills")
    result = await acquirer.acquire("write a sort function", ["code_gen"])

    assert result.found is True
    assert result.level == AcquisitionLevel.HISTORY
    assert "sorting algorithm" in result.content


@pytest.mark.asyncio
async def test_l1_history_no_match():
    """L1 should return not found when Qdrant has no matching data."""
    from common.vector import get_qdrant_client, ensure_collection

    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_empty")

    acquirer = SkillAcquirer(qdrant=client, collection_name="test_empty")
    with patch("agents.skill_acquirer.shutil.which", return_value=None):
        result = await acquirer.acquire("do something", ["skill_a"])

    assert result.found is False
    assert result.level in (AcquisitionLevel.HISTORY, AcquisitionLevel.WEB_SEARCH, AcquisitionLevel.EXTERNAL_SKILL)


@pytest.mark.asyncio
async def test_l2_web_search():
    """L2 should use mcporter search."""
    import json as json_mod
    acquirer = SkillAcquirer()

    mock_output = json_mod.dumps({
        "pages": [
            {"title": "Python Sort", "snippet": "How to sort in Python", "url": "https://example.com"},
        ],
    }).encode()

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(mock_output, b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
         patch("agents.skill_acquirer.shutil.which", return_value="/usr/bin/mcporter"):
        result = await acquirer.acquire("Python sorting algorithms", ["code_gen"])

    assert result.found is True
    assert result.level == AcquisitionLevel.WEB_SEARCH
    assert "Python Sort" in result.content


@pytest.mark.asyncio
async def test_l3_external_skill_approved():
    """L3 should generate and approve safe code."""
    mock_llm = AsyncMock()
    # First call: generate code
    # Second call: security review
    mock_llm.ainvoke.side_effect = [
        MagicMock(content='def add(a, b):\n    """Add two numbers."""\n    return a + b'),
        MagicMock(content='{"risk_level": "safe", "reason": "Pure function, no side effects"}'),
    ]

    acquirer = SkillAcquirer(llm=mock_llm)
    with patch("agents.skill_acquirer.shutil.which", return_value=None):
        result = await acquirer.acquire("add two numbers", ["math"])

    assert result.found is True
    assert result.level == AcquisitionLevel.EXTERNAL_SKILL
    assert "def add" in result.content
    assert result.metadata["security"] == "safe"


@pytest.mark.asyncio
async def test_l3_external_skill_needs_review():
    """L3 should flag suspicious code for human review."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = [
        MagicMock(content='import requests\ndef fetch(url):\n    return requests.get(url).text'),
        MagicMock(content='{"risk_level": "suspicious", "reason": "Makes network calls"}'),
    ]

    acquirer = SkillAcquirer(llm=mock_llm)
    with patch("agents.skill_acquirer.shutil.which", return_value=None):
        result = await acquirer.acquire("fetch a webpage", ["web"])

    assert result.found is False
    assert "NEEDS_HUMAN_REVIEW" in result.content
    assert result.metadata["needs_review"] is True


# === Security Review tests ===

def test_static_check_clean_code():
    findings = quick_static_check("def add(a, b):\n    return a + b")
    assert findings == []


def test_static_check_dangerous():
    code = "import subprocess\nos.system('rm -rf /')\neval(input())"
    findings = quick_static_check(code)
    assert len(findings) >= 2
    assert any("os.system" in f for f in findings)
    assert any("eval" in f for f in findings)


@pytest.mark.asyncio
async def test_review_safe_code_no_llm():
    """Without LLM, any non-empty code should be marked for human review."""
    verdict = await review_skill("def add(a, b): return a + b", llm=None)
    assert verdict.needs_human_review is True
    assert verdict.approved is False


@pytest.mark.asyncio
async def test_review_dangerous_code():
    """Code with many dangerous patterns should be rejected outright."""
    code = """
import subprocess
os.system('rm -rf /')
eval(user_input)
exec(code)
subprocess.run(['ls'])
"""
    verdict = await review_skill(code, llm=None)
    assert verdict.risk_level == RiskLevel.DANGEROUS
    assert verdict.approved is False


@pytest.mark.asyncio
async def test_review_empty_code():
    verdict = await review_skill("", llm=None)
    assert verdict.risk_level == RiskLevel.SAFE


@pytest.mark.asyncio
async def test_review_with_llm():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content='{"risk_level": "safe", "reason": "Simple math function"}'
    )
    verdict = await review_skill("def add(a, b): return a + b", llm=mock_llm)
    assert verdict.risk_level == RiskLevel.SAFE
    assert verdict.approved is True
