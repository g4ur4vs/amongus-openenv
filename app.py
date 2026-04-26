from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amongus_env.eval_suite import run_eval_suite  # noqa: E402
from amongus_env.golden_episode import run_golden_episode  # noqa: E402


def run_baseline_eval() -> str:
    return json.dumps(run_eval_suite(), indent=2)


def run_golden_trace() -> str:
    return json.dumps(run_golden_episode(), indent=2)


with gr.Blocks(title="Among Us OpenEnv Eval") as demo:
    gr.Markdown(
        "# Among Us OpenEnv Eval\n"
        "Deterministic social-deduction eval for the OpenEnv MVP."
    )
    with gr.Row():
        eval_button = gr.Button("Run Baseline Eval")
        trace_button = gr.Button("Run Golden Trace")
    output = gr.Code(language="json", label="Output")

    eval_button.click(run_baseline_eval, outputs=output)
    trace_button.click(run_golden_trace, outputs=output)


if __name__ == "__main__":
    demo.launch()
