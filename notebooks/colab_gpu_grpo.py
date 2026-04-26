# %% [markdown]
# # Among Us GRPO GPU Smoke
#
# Run in Google Colab with `Runtime > Change runtime type > T4 GPU`.
#
# Default model: `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, the smallest
# practical public model for proving the TRL GRPO trainer path. For a tiny but
# more realistic instruct model, switch `MODEL_ID` to
# `HuggingFaceTB/SmolLM2-135M-Instruct` and expect a slower run.
#
# This proves training plumbing. It is not a useful Among Us policy yet.

# %%
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(
    command: str,
    *,
    quiet: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {command}")
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    if not quiet and result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    if not quiet and result.stderr:
        print(result.stderr)
    return result


# %%
run("nvidia-smi")

# %%
REPO_URL = "https://github.com/g4ur4vs/amongus-openenv.git"
BRANCH = "main"
BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
WORKDIR = BASE_DIR / "amongus-openenv"

BASE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)
if WORKDIR.exists():
    run(f"rm -rf {shlex.quote(str(WORKDIR))}", cwd=BASE_DIR)
run(
    "git clone "
    f"--depth 1 --branch {shlex.quote(BRANCH)} "
    f"{shlex.quote(REPO_URL)} "
    f"{shlex.quote(str(WORKDIR))}",
    cwd=BASE_DIR,
)
os.chdir(WORKDIR)
print("cwd:", Path.cwd())

# %%
run(f"{sys.executable} -m pip install -U pip")
run(f"{sys.executable} -m pip install -e '.[training]'")

# %%
MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
# MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_DIR = "outputs/colab-grpo-tiny"

run(f"{sys.executable} -m amongus_env.eval_suite > baseline_eval.json")

# %%
train_command = (
    f"{sys.executable} -m amongus_env.grpo_train "
    "--construct-trainer --train --save-trained-model --allow-hub-model "
    "--reward-mode env_rollout "
    f"--model-id {MODEL_ID} "
    f"--output-dir {OUTPUT_DIR}"
)
run(f"{train_command} > rl_train.json")

# %%
status_path = Path(OUTPUT_DIR)
print("output_dir_exists:", status_path.exists())
print("output_files:", sorted(path.name for path in status_path.glob("*"))[:20])
final_model_path = status_path / "final_model"
print("final_model_exists:", final_model_path.exists())
print("final_model_files:", sorted(path.name for path in final_model_path.glob("*"))[:20])

# %%
run(
    "printf '%s\n%s\n' "
    "'{\"type\": \"move\", \"room\": \"Electrical\"}' "
    "'{\"type\": \"complete_task\"}' "
    "> rl_completion_actions.jsonl"
)
run(
    f"{sys.executable} -m amongus_env.policy_eval "
    "--rl-completions-file rl_completion_actions.jsonl "
    "> policy_eval.json"
)
run(
    f"{sys.executable} -m amongus_env.training_report "
    "--train-json rl_train.json "
    "--policy-eval-json policy_eval.json "
    "> baseline_vs_rl_report.json"
)
print(Path("baseline_vs_rl_report.json").read_text())
