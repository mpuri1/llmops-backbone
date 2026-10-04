"""Per-project daily budgets and a spend ledger for the LiteLLM proxy.

Every request must name its project (request metadata "project", or the OpenAI "user" field). Before a call,
the project's spend for the current UTC day is compared with its budget in gateway/budgets.yaml; at or over
budget the call is refused with HTTP 429. After each successful call, one line is appended to the ledger
(LLMOPS_GATEWAY_LEDGER, default gateway/spend.jsonl): time, project, alias, model that answered, tokens, cost.
The ledger is reloaded at start-up, so budgets survive a restart without a database.

A call LiteLLM can't price (a model missing from its price table) is logged with cost_usd null, and the project's
calls are refused for the rest of the day: a budget that silently counts $0 isn't a budget. Give the model a price
in gateway/config.yaml (model_info: input_cost_per_token, output_cost_per_token).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

HERE = Path(__file__).resolve().parent


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class ProjectBudgets(CustomLogger):
    def __init__(self, budgets_path: Path | None = None, ledger_path: Path | None = None):
        super().__init__()
        self.budgets_path = budgets_path or HERE / "budgets.yaml"
        self.ledger_path = ledger_path or Path(os.environ.get("LLMOPS_GATEWAY_LEDGER", HERE / "spend.jsonl"))
        self.budgets = {str(k): float(v) for k, v in yaml.safe_load(self.budgets_path.read_text()).items()}
        self.spent: dict[tuple[str, str], float] = defaultdict(float)
        self.unpriced: set[tuple[str, str]] = set()  # (project, day) with a call that had no cost
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                r = json.loads(line)
                self._add(r["project"], r["time"][:10], r["cost_usd"])

    def _add(self, project: str, day: str, cost: float | None) -> None:
        if cost is None:
            self.unpriced.add((project, day))
        else:
            self.spent[(project, day)] += cost

    def budget(self, project: str) -> float:
        return self.budgets.get(project, self.budgets.get("_default", 0.0))

    @staticmethod
    def project_of(data: dict) -> str | None:
        meta = data.get("metadata") or {}
        return meta.get("project") or data.get("user")

    def check(self, project: str | None) -> None:
        """Raise when a call can't go ahead: no project, or the project's budget for today is spent."""
        if not project:
            raise HTTPException(status_code=400, detail="name the calling project in metadata.project or user")
        if (project, _today()) in self.unpriced:
            raise HTTPException(
                status_code=429,
                detail=f"a call for {project} today had no price, so its budget can't be enforced; "
                "add model_info prices for that model in gateway/config.yaml",
            )
        spent, budget = self.spent[(project, _today())], self.budget(project)
        if spent >= budget:
            raise HTTPException(
                status_code=429, detail=f"daily budget for {project} spent: ${spent:.4f} of ${budget:.2f}"
            )

    def record(self, project: str, alias: str, model: str, usage: dict, cost: float | None) -> dict:
        row = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "project": project,
            "alias": alias,
            "model": model,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost_usd": None if cost is None else round(cost, 8),
        }
        self._add(project, row["time"][:10], cost)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        project = self.project_of(data)
        self.check(project)
        data.setdefault("metadata", {})["llmops_project"] = project  # carried to the success hook
        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        meta = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        project = meta.get("llmops_project") or "unknown"
        usage = getattr(response_obj, "usage", None)
        usage = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
        alias = meta.get("model_group") or kwargs.get("model", "")
        cost = kwargs.get("response_cost")  # None when LiteLLM has no price for the model
        self.record(project, alias, kwargs.get("model", ""), usage, None if cost is None else float(cost))


budgets = ProjectBudgets()
