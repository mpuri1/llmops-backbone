"""Send one traced agent step with a nested model-call span to the local Phoenix.

    phoenix serve            # in another terminal (uv run --with arize-phoenix phoenix serve)
    uv run python examples/trace_smoke.py

No model is called: the LLM span records the attributes a real call would carry.
"""

from llmops_kit import llm_span, setup_tracing, traced

TEMPLATE = "Map the column {column} of table {table} to one ontology property."


@traced(kind="TOOL")
def profile_table(table: str) -> dict:
    return {"table": table, "columns": 73}


@traced(name="map_columns", kind="AGENT")
def map_columns(table: str) -> str:
    profile_table(table)
    with llm_span(model="stub-model", provider="none", prompt_template=TEMPLATE, prompt_name="map_column") as call:
        call.record_usage(input_tokens=0, output_tokens=0)
    return "ok"


if __name__ == "__main__":
    provider = setup_tracing("llmops-smoke", batch=False)
    map_columns("nfip_claims")
    provider.force_flush()
    print("sent 3 spans to Phoenix")
