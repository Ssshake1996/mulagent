"""Agent registry — maintains agent metadata, skills, and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.config import CONFIG_DIR


@dataclass
class AgentMeta:
    id: str
    name: str
    description: str
    skills: list[str] = field(default_factory=list)
    agent_type: str = "thinker"  # thinker | retriever | executor
    model: str = "qwen3.5-plus"
    priority: int = 10
    success_rate: float = 1.0
    total_runs: int = 0


class AgentRegistry:
    """Central registry for all available agents."""

    def __init__(self):
        self._agents: dict[str, AgentMeta] = {}

    def register(self, agent: AgentMeta) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentMeta | None:
        return self._agents.get(agent_id)

    def list_all(self) -> list[AgentMeta]:
        return list(self._agents.values())

    def select_by_type(self, agent_type: str) -> AgentMeta | None:
        """Select the best agent of a given type (thinker/retriever/executor)."""
        candidates = [a for a in self._agents.values() if a.agent_type == agent_type]
        if not candidates:
            return None
        # Pick highest priority (lowest number) with best success rate
        candidates.sort(key=lambda a: (a.priority, -a.success_rate))
        return candidates[0]

    def find_by_skill(self, skill: str) -> list[AgentMeta]:
        """Find agents that have a specific skill."""
        return [a for a in self._agents.values() if skill in a.skills]

    def select_best(self, required_skills: list[str]) -> AgentMeta | None:
        """Legacy: select agent by skill matching. Prefer select_by_type for new code."""
        candidates = []
        for agent in self._agents.values():
            matched = set(required_skills) & set(agent.skills)
            if not matched:
                continue
            score = (
                len(matched) / len(required_skills) * 0.5
                + agent.success_rate * 0.3
                + (1.0 / agent.priority) * 0.2
            )
            candidates.append((score, agent))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def update_stats(self, agent_id: str, success: bool) -> None:
        """Update agent success rate after a task run."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        agent.total_runs += 1
        # Exponential moving average
        alpha = 0.1
        agent.success_rate = agent.success_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha


def load_registry(config_path: Path | None = None) -> AgentRegistry:
    """Load agent registry from YAML config."""
    path = config_path or CONFIG_DIR / "agents.yaml"
    registry = AgentRegistry()

    if not path.exists():
        return registry

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for agent_id, cfg in data.get("agents", {}).items():
        registry.register(
            AgentMeta(
                id=agent_id,
                name=cfg.get("name", agent_id),
                description=cfg.get("description", ""),
                skills=cfg.get("skills", []),
                agent_type=cfg.get("type", "thinker"),
                model=cfg.get("model", "qwen3.5-plus"),
                priority=cfg.get("priority", 10),
            )
        )
    return registry
