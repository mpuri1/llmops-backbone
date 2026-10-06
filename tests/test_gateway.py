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


def test_budget_hook_refuses_a_project_after_a_call_it_could_not_price(tmp_path):
    hook = load_hook()
    budgets = tmp_path / "budgets.yaml"
    budgets.write_text("_default: 5.0\n")
    ledger = tmp_path / "spend.jsonl"
    b = hook.ProjectBudgets(budgets, ledger)

    b.record("p3", "smart", "bedrock/new-model", {"prompt_tokens": 10, "completion_tokens": 5}, None)
    assert json.loads(ledger.read_text())["cost_usd"] is None  # not logged as $0
    with pytest.raises(hook.HTTPException) as e:
        b.check("p3")
    assert e.value.status_code == 429 and "model_info" in e.value.detail
    with pytest.raises(hook.HTTPException):
        hook.ProjectBudgets(budgets, ledger).check("p3")  # still refused after a restart
    b.check("other")


class FakeGateway(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["metadata"]["project"] == "broke":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error": "daily budget for broke spent"}')
            return
        payload = {
            "model": "fast",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }
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


def test_bedrock_proxy_url_and_signing():
    pytest.importorskip("botocore")
    from botocore.credentials import Credentials

    from llmops_kit.bedrock_proxy import sign_headers, upstream_url

    assert upstream_url("us-east-2", "/v1/chat/completions") == (
        "https://bedrock-mantle.us-east-2.api.aws/v1/chat/completions"
    )
    assert upstream_url("us-east-2", "/chat/completions").endswith("/v1/chat/completions")
    url = upstream_url("us-east-2", "/v1/chat/completions")
    headers = sign_headers(Credentials("AKIDEXAMPLE", "secret", "token"), "us-east-2", "POST", url, b"{}")
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
    assert headers["X-Amz-Security-Token"] == "token"


def test_bedrock_proxy_forwards_signed_requests(monkeypatch):
    pytest.importorskip("botocore")
    from botocore.credentials import Credentials

    from llmops_kit import bedrock_proxy

    seen = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    import urllib.request

    real_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout=None):
        if "bedrock-mantle" not in request.full_url:
            return real_urlopen(request, timeout=timeout)
        seen["url"], seen["body"], seen["auth"] = request.full_url, request.data, request.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(bedrock_proxy.urllib.request, "urlopen", fake_urlopen)
    handler = bedrock_proxy.make_handler(lambda: Credentials("AKIDEXAMPLE", "secret"), "us-east-2")
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    body = json.dumps({"model": "x"}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=body, method="POST"
    )
    try:
        with real_urlopen(request, timeout=10) as response:
            assert json.loads(response.read()) == {"ok": True}
    finally:
        server.server_close()
    assert seen["url"] == "https://bedrock-mantle.us-east-2.api.aws/v1/chat/completions"
    assert seen["body"] == body and seen["auth"].startswith("AWS4-HMAC-SHA256")
