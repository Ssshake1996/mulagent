"""Lightweight skill security review — LLM判断 + 人工兜底。

安全策略：
1. LLM 自动审查代码片段，判断风险等级（safe / suspicious / dangerous）
2. safe → 直接通过
3. suspicious → 标记待人工审核，暂不执行
4. dangerous → 直接拒绝
5. 无 LLM 时 → 保守策略，全部标记待人工审核
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


@dataclass
class SecurityVerdict:
    risk_level: RiskLevel
    reason: str
    needs_human_review: bool
    approved: bool


# 明确禁止的模式（不需要 LLM 就能判断）
DANGEROUS_PATTERNS = [
    "os.system(",
    "subprocess.call(",
    "subprocess.Popen(",
    "subprocess.run(",
    "__import__(",
    "eval(",
    "exec(",
    "shutil.rmtree(",
    "os.remove(",
    "os.rmdir(",
    "open(",  # 文件写入
]

# 危险 import
DANGEROUS_IMPORTS = [
    "import subprocess",
    "import shutil",
    "import ctypes",
    "import socket",
    "from os import system",
]


def quick_static_check(code: str) -> list[str]:
    """快速静态检查，返回发现的危险模式列表。"""
    findings = []
    for pattern in DANGEROUS_PATTERNS:
        if pattern in code:
            findings.append(f"Dangerous call: {pattern.rstrip('(')}")
    for imp in DANGEROUS_IMPORTS:
        if imp in code:
            findings.append(f"Dangerous import: {imp}")
    return findings


async def review_skill(
    code: str,
    source: str = "unknown",
    llm=None,
) -> SecurityVerdict:
    """审查一段外部 Skill 代码的安全性。

    Args:
        code: 要审查的代码片段
        source: 代码来源（如 URL、包名）
        llm: LLM 客户端，用于智能判断
    """
    # Step 1: 快速静态检查
    static_findings = quick_static_check(code)

    if not code.strip():
        return SecurityVerdict(
            risk_level=RiskLevel.SAFE,
            reason="Empty code",
            needs_human_review=False,
            approved=False,
        )

    # 明显的危险模式 → 直接拒绝
    if any("Dangerous" in f for f in static_findings) and len(static_findings) >= 3:
        return SecurityVerdict(
            risk_level=RiskLevel.DANGEROUS,
            reason=f"Multiple dangerous patterns: {'; '.join(static_findings[:5])}",
            needs_human_review=False,
            approved=False,
        )

    # Step 2: LLM 智能审查
    if llm is not None:
        try:
            verdict = await _llm_review(code, source, static_findings, llm)
            return verdict
        except Exception as e:
            logger.warning("LLM security review failed: %s, falling back to conservative", e)

    # Step 3: 无 LLM 或 LLM 失败 → 保守策略
    if static_findings:
        return SecurityVerdict(
            risk_level=RiskLevel.SUSPICIOUS,
            reason=f"Static check findings: {'; '.join(static_findings)}. No LLM available for deep analysis.",
            needs_human_review=True,
            approved=False,
        )

    return SecurityVerdict(
        risk_level=RiskLevel.SUSPICIOUS,
        reason="No LLM available for review, marking for human check",
        needs_human_review=True,
        approved=False,
    )


async def _llm_review(
    code: str,
    source: str,
    static_findings: list[str],
    llm,
) -> SecurityVerdict:
    """用 LLM 审查代码安全性。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    static_info = f"\nStatic analysis findings: {static_findings}" if static_findings else ""

    messages = [
        SystemMessage(content=(
            "You are a code security reviewer. Analyze the given code snippet and assess its safety.\n"
            "Respond with ONLY a JSON object:\n"
            '{"risk_level": "safe|suspicious|dangerous", "reason": "brief explanation"}'
        )),
        HumanMessage(content=(
            f"Source: {source}\n"
            f"Code:\n```\n{code[:2000]}\n```\n"  # 截断过长代码
            f"{static_info}"
        )),
    ]

    response = await llm.ainvoke(messages)
    result = json.loads(response.content)
    risk = RiskLevel(result.get("risk_level", "suspicious"))
    reason = result.get("reason", "")

    return SecurityVerdict(
        risk_level=risk,
        reason=reason,
        needs_human_review=(risk == RiskLevel.SUSPICIOUS),
        approved=(risk == RiskLevel.SAFE),
    )
