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

## Verify

```bash
python3 -m pytest
```

## Golden Episode

Run the deterministic judge-facing trace:

```bash
PYTHONPATH=src python3 -m amongus_env.golden_episode
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

## Eval Suite

Run the pass/fail eval JSON:

```bash
PYTHONPATH=src python3 -m amongus_env.eval_suite
```

The eval currently runs six deterministic scenarios: golden false alibi, invalid
movement, crewmate task route, meeting pass/no-majority, impostor parity win,
and kill cooldown blocking.

## Hugging Face Space

The repo includes a Gradio `app.py` for a Space. It exposes buttons for the
baseline eval and the full golden trace.

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
