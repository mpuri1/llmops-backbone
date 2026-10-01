import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llmops_kit.gateway import GatewayError, chat

ROOT = Path(__file__).resolve().parents[1]


def load_hook():
    pytest.importorskip("litellm")
    spec = importlib.util.spec_from_file_location("budget_hook", ROOT / "gateway" / "budget_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_hook_refuses_unnamed_and_over_budget_projects_and_survives_restart(tmp_path):
    hook = load_hook()
    budgets = tmp_path / "budgets.yaml"
    budgets.write_text("_default: 0.01\np2: 0.05\n")
    ledger = tmp_path / "spend.jsonl"
    b = hook.ProjectBudgets(budgets, ledger)

    with pytest.raises(hook.HTTPException) as e:
        b.check(None)
    assert e.value.status_code == 400

    b.check("p2")  # nothing spent yet
    b.record("p2", "fast", "gemini/gemini-3.5-flash-lite", {"prompt_tokens": 10, "completion_tokens": 5}, 0.06)
    with pytest.raises(hook.HTTPException) as e:
        b.check("p2")
    assert e.value.status_code == 429 and "p2" in e.value.detail

    restarted = hook.ProjectBudgets(budgets, ledger)  # spend comes back from the ledger
    with pytest.raises(hook.HTTPException):
        restarted.check("p2")
    restarted.check("other")  # a project without an entry uses _default and has spent nothing
    assert json.loads(ledger.read_text().splitlines()[0])["model"] == "gemini/gemini-3.5-flash-lite"


class FakeGateway(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["metadata"]["project"] == "broke":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error": "daily budget for broke spent"}')
            return
        payload = {"model": "fast", "choices": [{"message": {"content": "hello"},
                   "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        self.send_response(200)
        self.send_header("x-litellm-response-cost", "0.000012")
        self.send_header("x-litellm-model-name", "gemini/gemini-2.5-flash")  # answered by the fallback
        self.send_header("x-litellm-attempted-fallbacks", "1")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def gateway_url():
    server = HTTPServer(("127.0.0.1", 0), FakeGateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_client_returns_text_tokens_cost_and_the_model_that_answered(gateway_url):
    reply = chat([{"role": "user", "content": "hi"}], model="fast", project="p2", base_url=gateway_url)
    assert (reply.text, reply.model) == ("hello", "gemini/gemini-2.5-flash")
    assert (reply.input_tokens, reply.output_tokens) == (7, 2)
    assert reply.cost_usd == pytest.approx(0.000012) and reply.fallbacks == 1


def test_client_requires_a_project_and_surfaces_budget_refusals(gateway_url, monkeypatch):
    monkeypatch.delenv("LLMOPS_PROJECT", raising=False)
    with pytest.raises(ValueError):
        chat([{"role": "user", "content": "hi"}], base_url=gateway_url)
    with pytest.raises(GatewayError) as e:
        chat([{"role": "user", "content": "hi"}], project="broke", base_url=gateway_url)
    assert e.value.status == 429
