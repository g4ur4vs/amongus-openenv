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
import subprocess
import sys
from pathlib import Path


def run(command: str) -> None:
    print(f"$ {command}")
    subprocess.run(command, shell=True, check=True)


# %%
run("nvidia-smi")

# %%
REPO_URL = "https://github.com/g4ur4vs/amongus-openenv.git"
BRANCH = "main"
WORKDIR = Path("/content/amongus-openenv")

if WORKDIR.exists():
    run(f"rm -rf {WORKDIR}")
run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {WORKDIR}")
os.chdir(WORKDIR)
print("cwd:", Path.cwd())

# %%
run(f"{sys.executable} -m pip install -U pip")
run(f"{sys.executable} -m pip install -e '.[training]'")

# %%
MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
# MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_DIR = "outputs/colab-grpo-tiny"

run(f"{sys.executable} -m amongus_env.eval_suite")

# %%
train_command = (
    f"{sys.executable} -m amongus_env.grpo_train "
    "--construct-trainer --train --allow-hub-model "
    f"--model-id {MODEL_ID} "
    f"--output-dir {OUTPUT_DIR}"
)
run(train_command)

# %%
status_path = Path(OUTPUT_DIR)
print("output_dir_exists:", status_path.exists())
print("output_files:", sorted(path.name for path in status_path.glob("*"))[:20])
