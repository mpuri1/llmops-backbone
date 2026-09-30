"""Trace one real Gemini call into local Phoenix.

    uv run --with google-genai python examples/trace_gemini.py

Needs GOOGLE_API_KEY in the environment (or in a .env file here). Uses the
cheapest Flash-Lite model and a one-line prompt.
"""

import os

from dotenv import load_dotenv
from google import genai

from llmops_kit import llm_span, setup_tracing, traced

load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
TEMPLATE = (
    "An insurance dataset has a column named '{column}'. "
    "In at most eight words, what does it most likely hold?"
)


@traced(name="describe_column", kind="AGENT")
def describe_column(client: genai.Client, column: str) -> str:
    with llm_span(model=MODEL, provider="gcp.gemini", prompt_template=TEMPLATE, prompt_name="describe_column") as call:
        response = client.models.generate_content(model=MODEL, contents=TEMPLATE.format(column=column))
        usage = response.usage_metadata
        call.record_usage(input_tokens=usage.prompt_token_count, output_tokens=usage.candidates_token_count)
        call.record_response(model=response.model_version,
                             finish_reason=response.candidates[0].finish_reason.name.lower())
    return response.text.strip()


if __name__ == "__main__":
    provider = setup_tracing("llmops-gemini-smoke", batch=False)
    print(describe_column(genai.Client(), "amountPaidOnBuildingClaim"))
    provider.force_flush()
