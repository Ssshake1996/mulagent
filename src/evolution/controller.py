"""Evolution Controller — orchestrates the full self-evolution cycle.

Entry point for all evolution operations:
  - diagnose: analyze system weaknesses
  - prescribe: generate improvement proposals
  - apply: execute improvements safely
  - absorb: integrate external projects
  - evolve: full cycle (diagnose → prescribe → apply)

Safety tiers:
  - Auto-apply: parameter tuning, experience quality adjustments
  - Needs confirmation: prompt changes, new skills/tools, knowledge updates
  - Never auto-apply: core source code changes
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Safety classification
_AUTO_SAFE_TYPES = {"tune_params"}
_CONFIRM_TYPES = {"prompt_refine", "add_skill", "add_tool", "update_knowledge"}
_NEVER_AUTO_TYPES = {"modify_source"}


@dataclass
class EvolutionReport:
    """Result of an evolution cycle."""
    mode: str = ""           # "auto", "propose", "absorb"
    diagnosis_summary: str = ""
    proposed: list[dict] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "diagnosis_summary": self.diagnosis_summary,
            "proposed": self.proposed,
            "applied": self.applied,
            "skipped": self.skipped,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        lines = [f"=== Evolution Report ({self.mode}) — {self.timestamp} ==="]
        if self.diagnosis_summary:
            lines.append(f"\n{self.diagnosis_summary}")
        if self.proposed:
            lines.append(f"\nProposed ({len(self.proposed)}):")
            for p in self.proposed:
                lines.append(f"  [{p.get('type')}] {p.get('target')} — {p.get('reason', '')[:60]}")
        if self.applied:
            lines.append(f"\nApplied ({len(self.applied)}):")
            for a in self.applied:
                status = "✓" if a.get("success") else "✗"
                lines.append(f"  {status} {a.get('evolution_type')}: {a.get('message', '')[:60]}")
        if self.skipped:
            lines.append(f"\nSkipped ({len(self.skipped)}) — needs confirmation:")
            for s in self.skipped:
                lines.append(f"  ? [{s.get('type')}] {s.get('target')} — {s.get('reason', '')[:60]}")
        if self.errors:
            lines.append(f"\nErrors: {'; '.join(self.errors)}")
        return "\n".join(lines)


class EvolutionController:
    """Orchestrates the self-evolution pipeline."""

    def __init__(self):
        from evolution.diagnostician import Diagnostician
        from evolution.prescriber import Prescriber
        from evolution.applier import Applier
        from evolution.absorber import Absorber

        self.diagnostician = Diagnostician()
        self.prescriber = Prescriber()
        self.applier = Applier()
        self.absorber = Absorber()

    async def evolve(
        self,
        mode: str = "propose",
        days: int = 7,
        min_confidence: float = 0.5,
        llm=None,
    ) -> EvolutionReport:
        """Run a full evolution cycle.

        Args:
            mode: "auto" = apply safe changes automatically,
                  "propose" = only suggest (default),
                  "full" = apply all (with confirmation).
            days: Number of days of history to analyze.
            min_confidence: Minimum confidence threshold for proposals.
            llm: LLM instance for intelligent analysis.

        Returns:
            EvolutionReport with proposals and results.
        """
        report = EvolutionReport(
            mode=mode,
            timestamp=datetime.now(timezone.utc).isoformat()[:19],
        )

        # Step 1: Diagnose
        try:
            diagnosis = await self.diagnostician.diagnose(days=days)
            report.diagnosis_summary = diagnosis.summary()
        except Exception as e:
            report.errors.append(f"Diagnosis failed: {e}")
            logger.exception("Evolution diagnosis failed")
            return report

        # Step 2: Prescribe
        try:
            evolutions = await self.prescriber.prescribe(
                diagnosis.to_dict(), llm=llm
            )
            # Filter by confidence
            evolutions = [e for e in evolutions if e.confidence >= min_confidence]
        except Exception as e:
            report.errors.append(f"Prescription failed: {e}")
            logger.exception("Evolution prescription failed")
            return report

        if not evolutions:
            report.proposed = []
            return report

        # Step 3: Apply based on mode
        for evo in evolutions:
            report.proposed.append(evo.to_dict())

        if mode == "propose":
            # Don't apply anything, just report
            return report

        auto_evolutions = []
        confirm_evolutions = []

        for evo in evolutions:
            if evo.type in _AUTO_SAFE_TYPES:
                auto_evolutions.append(evo)
            elif evo.type in _CONFIRM_TYPES:
                confirm_evolutions.append(evo)
            # _NEVER_AUTO_TYPES are always skipped

        # Auto-apply safe changes
        if auto_evolutions:
            try:
                results = await self.applier.apply(auto_evolutions)
                for r in results:
                    report.applied.append({
                        "evolution_type": r.evolution_type,
                        "target": r.target,
                        "success": r.success,
                        "message": r.message,
                    })
            except Exception as e:
                report.errors.append(f"Auto-apply failed: {e}")

        if mode == "auto":
            # In auto mode, skip confirm-required changes
            for evo in confirm_evolutions:
                report.skipped.append(evo.to_dict())
        elif mode == "full":
            # In full mode, apply everything
            if confirm_evolutions:
                try:
                    results = await self.applier.apply(confirm_evolutions)
                    for r in results:
                        report.applied.append({
                            "evolution_type": r.evolution_type,
                            "target": r.target,
                            "success": r.success,
                            "message": r.message,
                        })
                except Exception as e:
                    report.errors.append(f"Apply failed: {e}")

        # Log evolution record
        self._save_evolution_log(report)
        return report

    async def absorb_project(
        self,
        git_url: str,
        auto_apply: bool = False,
        llm=None,
    ) -> EvolutionReport:
        """Absorb an external project's capabilities.

        Args:
            git_url: Git repository URL to analyze.
            auto_apply: If True, apply generated evolutions automatically.
            llm: LLM instance for intelligent analysis.

        Returns:
            EvolutionReport with proposals and optionally applied results.
        """
        report = EvolutionReport(
            mode="absorb",
            timestamp=datetime.now(timezone.utc).isoformat()[:19],
        )

        try:
            evolutions = await self.absorber.absorb(git_url, llm=llm)
        except Exception as e:
            report.errors.append(f"Absorb failed: {e}")
            logger.exception("Project absorption failed")
            return report

        for evo in evolutions:
            report.proposed.append(evo.to_dict())

        if auto_apply and evolutions:
            try:
                results = await self.applier.apply(evolutions)
                for r in results:
                    report.applied.append({
                        "evolution_type": r.evolution_type,
                        "target": r.target,
                        "success": r.success,
                        "message": r.message,
                    })
            except Exception as e:
                report.errors.append(f"Apply failed: {e}")

        self._save_evolution_log(report)
        return report

    async def diagnose_only(self, days: int = 7) -> str:
        """Run diagnosis only and return the summary."""
        diagnosis = await self.diagnostician.diagnose(days=days)
        return diagnosis.summary()

    async def apply_evolution(
        self,
        evolution_dict: dict[str, Any],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Apply a single evolution from its dict representation.

        Used for confirming previously proposed evolutions.
        """
        from evolution.prescriber import Evolution
        evo = Evolution(
            type=evolution_dict["type"],
            target=evolution_dict["target"],
            reason=evolution_dict.get("reason", ""),
            patch=evolution_dict.get("patch", ""),
            priority=evolution_dict.get("priority", 2),
            confidence=evolution_dict.get("confidence", 0.5),
        )
        results = await self.applier.apply([evo], dry_run=dry_run)
        if results:
            r = results[0]
            return {
                "success": r.success,
                "message": r.message,
                "backup": r.backup_path,
            }
        return {"success": False, "message": "No result"}

    def _save_evolution_log(self, report: EvolutionReport) -> None:
        """Persist evolution report to file for history tracking."""
        try:
            from common.config import DATA_DIR
            log_dir = DATA_DIR / "evolution_logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            ts = report.timestamp.replace(":", "-")
            log_path = log_dir / f"evolution_{ts}.json"
            log_path.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            )
            logger.info("Evolution log saved: %s", log_path)
        except Exception as e:
            logger.debug("Failed to save evolution log: %s", e)

    def list_evolution_logs(self, limit: int = 10) -> list[dict]:
        """List recent evolution logs."""
        try:
            from common.config import DATA_DIR
            log_dir = DATA_DIR / "evolution_logs"
            if not log_dir.exists():
                return []

            logs = sorted(log_dir.glob("evolution_*.json"), reverse=True)
            results = []
            for log_path in logs[:limit]:
                try:
                    data = json.loads(log_path.read_text())
                    results.append({
                        "file": log_path.name,
                        "timestamp": data.get("timestamp", ""),
                        "mode": data.get("mode", ""),
                        "proposed": len(data.get("proposed", [])),
                        "applied": len(data.get("applied", [])),
                    })
                except Exception:
                    continue
            return results
        except Exception:
            return []
