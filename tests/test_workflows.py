"""One complete cached-or-fresh smoke workflow acceptance path."""

import json

from thermotwin.config import load_config
from thermotwin.forecasting.registry import load_bundle
from thermotwin.workflows import prepare_project, prepared_paths


def test_smoke_workflow_has_all_models_experiments_and_report():
    run_dir,record=prepare_project("smoke",force=False)
    assert record["status"]=="complete"
    config=load_config(profile="smoke"); dataset,models,_=prepared_paths(config); bundle,metadata=load_bundle(models)
    assert dataset.is_file()
    assert set(bundle["classical"])=={"ridge","random_forest","gradient_boosting"}
    assert bundle["recurrent"].trained_epochs>=1
    assert bundle["action_residual"].model is not None
    summary=json.loads((run_dir/"summary.json").read_text(encoding="utf-8")); assert set(summary["experiments"])==set("123456")
    manifest=json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["split_ranges"])==5
    assert set(manifest["seeds"]["streams"])=={"normal","burst","skewed","hot_ambient","cooling_degradation"}
    assert "manifest.json" not in {item["artifact_id"] for item in manifest["artifacts"]}
    assert (run_dir/"report.html").is_file()
    assert all((run_dir/f"experiment_{n}_status.json").is_file() for n in range(1,7))
