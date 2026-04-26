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

The eval currently checks the golden false-alibi episode for label order, task
reward, meeting protocol state, false-alibi penalty, and bot-vote ejection.

## Hugging Face Space

The repo includes a Gradio `app.py` for a Space. It exposes buttons for the
baseline eval and the full golden trace.
