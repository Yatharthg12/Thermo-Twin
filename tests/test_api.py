"""Cross-layer API validation, read-only GET behavior, control semantics, jobs, and downloads."""

import json
import time
from pathlib import Path

import yaml

from thermotwin.web import create_app


def _app(tmp_path):
    override=tmp_path/"test.yaml"; override.write_text(yaml.safe_dump({"project":{"artifacts_dir":str(tmp_path/"artifacts"),"reports_dir":str(tmp_path/"reports")}}),encoding="utf-8")
    return create_app(str(override),profile="smoke",testing=True)


def test_health_unknown_ids_and_method_semantics(tmp_path):
    client=_app(tmp_path).test_client()
    assert client.get("/api/health").status_code==200
    assert client.get("/api/simulations/nope").status_code==404
    assert client.get("/api/jobs").get_json()=={"jobs":[]}
    assert client.post("/api/jobs",json={"profile":"research"}).status_code==400


def test_get_is_read_only_and_pause_step_reset_work(tmp_path):
    app=_app(tmp_path); client=app.test_client(); created=client.post("/api/simulations",json={"scenario":"normal","controller":"fixed"}).get_json(); identifier=created["simulation_id"]
    first=client.get(f"/api/simulations/{identifier}").get_json(); second=client.get(f"/api/simulations/{identifier}").get_json(); assert first["snapshot"]["time_s"]==second["snapshot"]["time_s"]==0
    stepped=client.post(f"/api/simulations/{identifier}/control",json={"command":"step"}).get_json(); assert stepped["snapshot"]["time_s"]==60
    reset=client.post(f"/api/simulations/{identifier}/control",json={"command":"reset"}).get_json(); assert reset["snapshot"]["time_s"]==0
    app.extensions["thermotwin"]["simulations"].shutdown(); app.extensions["thermotwin"]["jobs"].shutdown()


def test_nan_validation_and_history_bound(tmp_path):
    client=_app(tmp_path).test_client(); response=client.post("/api/simulations",data='{"speed": NaN}',content_type="application/json"); assert response.status_code==400
    created=client.post("/api/simulations",json={"controller":"fixed"}).get_json(); assert client.get(f"/api/simulations/{created['simulation_id']}/history?limit=999999").status_code==400


def test_job_cancellation_is_cooperative(tmp_path):
    app=_app(tmp_path); manager=app.extensions["thermotwin"]["jobs"]
    def runner(profile,log,cancel):
        for index in range(5): log(f"step {index}")
        while not cancel(): time.sleep(.01)
        raise InterruptedError
    manager.runner=runner; client=app.test_client(); job=client.post("/api/jobs",json={"profile":"smoke"}).get_json(); time.sleep(.03); response=client.post(f"/api/jobs/{job['job_id']}/cancel"); assert response.status_code==200
    listed=client.get("/api/jobs").get_json()["jobs"]; assert listed[0]["job_id"]==job["job_id"] and 0 < listed[0]["progress"] < 100
    for _ in range(50):
        status=client.get(f"/api/jobs/{job['job_id']}").get_json()["status"]
        if status=="cancelled": break
        time.sleep(.01)
    assert status=="cancelled"; manager.shutdown(); app.extensions["thermotwin"]["simulations"].shutdown()


def test_scoped_artifact_download_blocks_traversal(tmp_path):
    app=_app(tmp_path); root=tmp_path/"reports"/"run-1"; root.mkdir(parents=True); (root/"manifest.json").write_text(json.dumps({"status":"complete","profile":"smoke"}),encoding="utf-8"); (root/"summary.json").write_text("{}",encoding="utf-8")
    client=app.test_client(); assert client.get("/api/runs/run-1/artifacts/summary.json").status_code==200
    assert client.get("/api/runs/run-1/artifacts/../test.yaml").status_code in {400,404}
