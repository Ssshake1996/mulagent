"""Tests for agent registry."""

from agents.registry import AgentMeta, AgentRegistry, load_registry
from common.config import CONFIG_DIR


def _make_agent(id: str, skills: list[str], priority: int = 1) -> AgentMeta:
    return AgentMeta(id=id, name=id, description="test", skills=skills, priority=priority)


def test_register_and_get():
    reg = AgentRegistry()
    agent = _make_agent("a1", ["code_gen"])
    reg.register(agent)
    assert reg.get("a1") is agent
    assert reg.get("nonexistent") is None


def test_list_all():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen"]))
    reg.register(_make_agent("a2", ["search"]))
    assert len(reg.list_all()) == 2


def test_find_by_skill():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen", "debug"]))
    reg.register(_make_agent("a2", ["search"]))
    found = reg.find_by_skill("code_gen")
    assert len(found) == 1
    assert found[0].id == "a1"


def test_find_by_skill_no_match():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen"]))
    assert reg.find_by_skill("dance") == []


def test_select_best_single_match():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen", "debug"], priority=1))
    reg.register(_make_agent("a2", ["search"], priority=2))
    best = reg.select_best(["code_gen"])
    assert best is not None
    assert best.id == "a1"


def test_select_best_multiple_skills():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen"], priority=1))
    reg.register(_make_agent("a2", ["code_gen", "debug"], priority=1))
    best = reg.select_best(["code_gen", "debug"])
    assert best is not None
    assert best.id == "a2"  # matches more skills


def test_select_best_no_match():
    reg = AgentRegistry()
    reg.register(_make_agent("a1", ["code_gen"]))
    assert reg.select_best(["dance"]) is None


def test_update_stats():
    reg = AgentRegistry()
    agent = _make_agent("a1", ["code_gen"])
    reg.register(agent)
    reg.update_stats("a1", success=True)
    assert agent.total_runs == 1
    assert agent.success_rate > 0


def test_load_registry_from_yaml():
    reg = load_registry(CONFIG_DIR / "agents.yaml")
    assert len(reg.list_all()) == 3
    thinker = reg.get("thinker")
    assert thinker is not None
    assert thinker.agent_type == "thinker"
    assert thinker.name == "Thinker"
    retriever = reg.get("retriever")
    assert retriever is not None
    assert retriever.agent_type == "retriever"
    executor = reg.get("executor")
    assert executor is not None
    assert executor.agent_type == "executor"
