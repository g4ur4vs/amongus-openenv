import json
from pathlib import Path

from amongus_env.training_report import (
    build_training_report,
    main,
    summarize_training_result,
)


def test_summarize_training_result_reads_saved_checkpoint(tmp_path) -> None:
    final_model = tmp_path / "final_model"
    final_model.mkdir()
    (final_model / "model.safetensors").write_text("weights")
    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "ok": True,
                "trainer_constructed": True,
                "trained": True,
                "saved_model_path": str(final_model),
                "train_result": "TrainOutput(global_step=1, training_loss=0.0)",
            }
        )
    )

    summary = summarize_training_result(train_json)

    assert summary["trained"] is True
    assert summary["checkpoint_saved"] is True
    assert summary["saved_model_path"] == str(final_model)


def test_training_report_is_honest_about_missing_policy_improvement(tmp_path) -> None:
    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "ok": True,
                "trainer_constructed": True,
                "trained": True,
                "saved_model_path": None,
                "train_result": "TrainOutput(global_step=1, training_loss=0.0)",
            }
        )
    )

    report = build_training_report(train_json)

    assert report["schema_version"] == 1
    assert report["baseline_env_eval"]["summary"]["scenarios"] == 10
    assert report["rl_training"]["trained"] is True
    assert report["baseline_vs_rl"]["policy_improvement_claimed"] is False
    assert report["baseline_vs_rl"]["score_delta"] is None
    assert "model-policy evaluator" in report["baseline_vs_rl"]["reason"]


def test_training_report_cli_prints_valid_json(tmp_path, capsys) -> None:
    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "ok": True,
                "trainer_constructed": True,
                "trained": True,
                "saved_model_path": None,
                "train_result": "TrainOutput(global_step=1, training_loss=0.0)",
            }
        )
    )

    main(["--train-json", str(train_json)])

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["rl_training"]["trained"] is True
