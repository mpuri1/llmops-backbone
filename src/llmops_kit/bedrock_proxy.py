"""Local SigV4-signing proxy for Bedrock's OpenAI-compatible endpoint (bedrock-mantle).

LiteLLM's OpenAI provider can only send a bearer key, but bedrock-mantle wants a SigV4 signature (service name
"bedrock") and, in the member account this project uses, only the management account's credentials work. This proxy
sits between the two: LiteLLM posts to http://127.0.0.1:<port>/v1/..., the proxy signs the same request with the
AWS profile and forwards it, and the response comes back unchanged.

    BEDROCK_PROFILE=<your-aws-profile> BEDROCK_REGION=us-east-2 uv run --extra bedrock python -m llmops_kit.bedrock_proxy

Settings (environment): BEDROCK_PROFILE (default "default"), BEDROCK_REGION (default "us-east-2"),
BEDROCK_PROXY_PORT (default 4010), BEDROCK_PROXY_HOST (default 127.0.0.1; keep it local, the proxy has no
authentication of its own). boto3 is imported lazily so the rest of the kit needs no AWS SDK.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICE = "bedrock"


def upstream_url(region: str, path: str) -> str:
    """The bedrock-mantle URL for a request path such as /v1/chat/completions (a leading /v1 is optional)."""
    path = "/" + path.lstrip("/")
    if not path.startswith("/v1/"):
        path = "/v1" + path
    return f"https://bedrock-mantle.{region}.api.aws{path}"


def sign_headers(credentials, region: str, method: str, url: str, body: bytes) -> dict[str, str]:
    """Headers (including Authorization) for a SigV4-signed request; `credentials` is a botocore Credentials."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    request = AWSRequest(method=method, url=url, data=body, headers={"Content-Type": "application/json"})
    SigV4Auth(credentials, SERVICE, region).add_auth(request)
    return dict(request.headers)


def make_handler(get_credentials, region: str, timeout: float = 300):
    class Handler(BaseHTTPRequestHandler):
        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            url = upstream_url(region, self.path)
            headers = sign_headers(get_credentials(), region, method, url, body)
            request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status, payload = response.status, response.read()
            except urllib.error.HTTPError as exc:
                status, payload = exc.code, exc.read()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            self._forward("POST")

        def do_GET(self):  # noqa: N802
            self._forward("GET")

        def log_message(self, format, *args):  # noqa: A002 - quiet: the gateway ledger is the record
            pass

    return Handler


def main() -> None:
    import boto3

    region = os.environ.get("BEDROCK_REGION", "us-east-2")
    session = boto3.Session(profile_name=os.environ.get("BEDROCK_PROFILE"), region_name=region)
    # fetched per request: botocore refreshes expiring credentials, and a frozen copy keeps the signature stable
    handler = make_handler(lambda: session.get_credentials().get_frozen_credentials(), region)
    host, port = os.environ.get("BEDROCK_PROXY_HOST", "127.0.0.1"), int(os.environ.get("BEDROCK_PROXY_PORT", "4010"))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"bedrock signing proxy on http://{host}:{port} -> bedrock-mantle.{region} (profile {session.profile_name})")
    server.serve_forever()


if __name__ == "__main__":
    main()
