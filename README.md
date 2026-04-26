---
title: Among Us OpenEnv Eval
emoji: 🕵️
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Among Us OpenEnv MVP

Deterministic Among Us-style environment for OpenEnv and later GRPO/RLVR training.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

For OpenEnv HTTP serving or future TRL experiments:

```bash
python3 -m pip install -e ".[dev,training]"
```

For the local Gradio Space app:

```bash
python3 -m pip install -e ".[space]"
```

## Verify

```bash
python3 -m pytest
```

## Golden Episode

Run the deterministic judge-facing trace:

```bash
amongus-golden-trace
```

The trace demonstrates:

- A crewmate moving to Electrical and completing a task for `+0.2`.
- A meeting with explicit `voting_open` / `meeting_turns_remaining` protocol state.
- A false alibi, `I was in MedBay`, verified against private location history.
- The RLVR hallucination penalty of `-1.0`.
- Bot votes ejecting red, the false claimant. This does not mean the impostor was caught.

This is intentionally a reproducible eval artifact, not full GRPO training.

TRL tool summaries include compact tails of `message_log`, `discussion_log`, and
`claims`; the full structured values remain available on the `Observation`.

Run the reasoning-log version of the same trace:

```bash
amongus-reasoning-trace
```

This adds optional `reasoning.raw`, parsed `reasoning.thought`, and
`verifications` fields without changing the core `Observation` schema.

## Eval Suite

Run the pass/fail eval JSON:

```bash
amongus-baseline-eval
```

The eval currently runs ten deterministic scenarios: golden false alibi, invalid
movement, crewmate task route, meeting pass/bot-majority, impostor parity win,
kill cooldown blocking, impostor fake task, vent claim verification, no-majority
meeting, and multi-impostor parity.

Run the deterministic Deception Elo metric:

```bash
amongus-deception-elo
```

This scores the false self-location claim resolution as deceiver-vs-assembly Elo:
the deceiver loses Elo when the false claimant is ejected, and gains Elo when a
different player is ejected.

Run a deterministic repeated-trace Deception Elo leaderboard:

```bash
amongus-deception-leaderboard --runs 5
```

## MVP Mechanics

The environment now includes a small but verifiable deception surface:

- `fake_task` lets impostors fake the current room's task for `0.0` reward without
  mutating crewmate task progress.
- `vent(room)` lets impostors move through a tiny vent graph.
- `Speak("I saw red vent")` is parsed as a verifiable `saw_vent` claim.
- `Speak("I accuse blue")` is parsed as a verifiable accusation; bots can follow
  true accusations when no higher-priority false verifiable claim exists.

## GRPO Training Skeleton

Run the opt-in GRPO wiring smoke without downloading a model:

```bash
amongus-grpo-smoke
```

If `trl` is installed through `.[training]`, the smoke imports `GRPOConfig` and
`GRPOTrainer`. Without the training extra, it still verifies the local
`AmongUsToolEnv` and RLVR reward hook.

Run the GRPO training dry-run:

```bash
amongus-grpo-train
```

This prints the base model id, GRPO config, prompt set size, environment factory
contract, and reward contract. It intentionally does not construct a trainer,
load a model, download weights, or claim training results.

Reward contract: `reward_from_game_state([env])` returns one dense RLVR reward
per `AmongUsToolEnv`, equal to that environment's latest `Observation.reward`.
Use `aggregation="episode_return"` to return the sum of step rewards since the
last reset.

Construct a real `GRPOTrainer` only with a local model directory:

```bash
amongus-grpo-train --construct-trainer --local-model /path/to/local/model
```

Run one explicit trainer step only when you are ready to load that local model:

```bash
amongus-grpo-train --construct-trainer --train --local-model /path/to/local/model
```

The command refuses to load the default hub model. This prevents accidental 7B
downloads or fake training claims on machines that do not have the model and
training stack prepared.

### Colab GPU Smoke

Use `notebooks/colab_gpu_grpo.py` in Google Colab with a T4 GPU. It clones this
repo, installs `.[training]`, verifies the eval suite, and runs one explicit
GRPO training step with the smallest public TRL smoke model:

```bash
python -m amongus_env.grpo_train \
  --construct-trainer \
  --train \
  --save-trained-model \
  --allow-hub-model \
  --model-id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
  --output-dir outputs/colab-grpo-tiny
```

For a tiny-but-real instruct model, change `MODEL_ID` in the Colab script to
`HuggingFaceTB/SmolLM2-135M-Instruct`.

## OpenEnv Server

Run the OpenEnv HTTP server locally after installing the package:

```bash
uvicorn amongus_env.openenv_server:app --host 0.0.0.0 --port 8000
```

Build and run the dedicated OpenEnv server image:

```bash
docker build -f Dockerfile.openenv -t amongus-openenv:latest .
docker run --rm -p 8000:8000 amongus-openenv:latest
```

The Gradio Space app is separate from the OpenEnv HTTP server.
The server uses upstream OpenEnv's app factory when that API is installed; with
the current PyPI `openenv` package it serves the same `/health`, `/reset`, and
`/step` contract through an in-repo FastAPI fallback.

## Hugging Face Space

The repo includes a Gradio `app.py` for a Space. It exposes buttons for the
baseline eval, golden trace, reasoning trace, Deception Elo, leaderboard, and
GRPO dry-run status.

Create and upload the Space after authenticating safely:

```bash
read -s HF_TOKEN
export HF_TOKEN
export HF_USERNAME=your-hf-username-or-org
python3 scripts/create_hf_space.py
unset HF_TOKEN HF_USERNAME
```

You can also set `HF_SPACE_ID=username_or_org/space_name` to control the exact
Space repo id. The helper avoids Hugging Face `whoami` unless no namespace is
provided, because that endpoint is heavily rate-limited.