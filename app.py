from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amongus_env.eval_suite import run_eval_suite  # noqa: E402
from amongus_env.golden_episode import run_golden_episode  # noqa: E402
from amongus_env.reasoning_trace import run_golden_reasoning_trace  # noqa: E402
from amongus_env.deception_elo import compute_deception_elo  # noqa: E402
from amongus_env.deception_leaderboard import run_deception_leaderboard  # noqa: E402
from amongus_env.grpo_train import run_grpo_dry_run  # noqa: E402


def run_baseline_eval() -> str:
    return json.dumps(run_eval_suite(), indent=2)


def run_golden_trace() -> str:
    return json.dumps(run_golden_episode(), indent=2)


def run_reasoning_trace() -> str:
    return json.dumps(run_golden_reasoning_trace(), indent=2)


def run_deception_elo() -> str:
    return json.dumps(compute_deception_elo(run_golden_episode()), indent=2)


def run_leaderboard() -> str:
    return json.dumps(run_deception_leaderboard(runs=5), indent=2)


def run_grpo_status() -> str:
    return json.dumps(run_grpo_dry_run(require_trl=False), indent=2)


with gr.Blocks(title="Among Us OpenEnv Eval", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Among Us OpenEnv Eval\n"
        "Deterministic social-deduction eval, verifiable reasoning traces, "
        "Deception Elo, and GRPO wiring status."
    )
    with gr.Tabs():
        with gr.Tab("Eval"):
            eval_button = gr.Button("Run Baseline Eval", variant="primary")
            eval_output = gr.Code(language="json", label="Eval JSON")
            eval_button.click(run_baseline_eval, outputs=eval_output)

        with gr.Tab("Traces"):
            with gr.Row():
                trace_button = gr.Button("Golden Trace", variant="primary")
                reasoning_button = gr.Button("Reasoning Trace")
            trace_output = gr.Code(language="json", label="Trace JSON")
            trace_button.click(run_golden_trace, outputs=trace_output)
            reasoning_button.click(run_reasoning_trace, outputs=trace_output)

        with gr.Tab("Deception Elo"):
            with gr.Row():
                elo_button = gr.Button("Single Elo Run", variant="primary")
                leaderboard_button = gr.Button("5-Run Repeated-Trace Leaderboard")
            elo_output = gr.Code(language="json", label="Elo JSON")
            elo_button.click(run_deception_elo, outputs=elo_output)
            leaderboard_button.click(run_leaderboard, outputs=elo_output)

        with gr.Tab("GRPO Status"):
            gr.Markdown(
                "This is a safe wiring check. It does not download weights, "
                "load a model, or claim training results."
            )
            grpo_button = gr.Button("Run GRPO Dry Run", variant="primary")
            grpo_output = gr.Code(language="json", label="GRPO Status JSON")
            grpo_button.click(run_grpo_status, outputs=grpo_output)


if __name__ == "__main__":
    demo.launch()
