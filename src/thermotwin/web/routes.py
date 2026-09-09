"""Dashboard page and validated same-origin JSON endpoints."""

from __future__ import annotations

from pathlib import Path
from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from thermotwin.config import project_path
from thermotwin.physics.workload import SCENARIOS
from thermotwin.services.artifacts import artifact_catalog
from thermotwin.services.simulation_service import CONTROLLERS
from thermotwin.web.validation import bounded_int, finite_number, valid_id


api = Blueprint("thermotwin", __name__)


def _services():
    return current_app.extensions["thermotwin"]


@api.get("/")
def index():
    return render_template("index.html")


@api.get("/favicon.svg")
def favicon():
    return current_app.send_static_file("assets/favicon.svg")


@api.get("/api/health")
def health():
    ext = _services()
    return jsonify({"status": "ok", "models_available": ext["bundle"] is not None, "profile": ext["config"]["project"]["profile"], "source_fingerprint": ext["source_fingerprint"]})


@api.get("/api/config")
def config():
    cfg, bundle = _services()["config"], _services()["bundle"]
    return jsonify({
        "profile": cfg["project"]["profile"], "topology": cfg["topology"], "thresholds": cfg["thresholds"],
        "scenarios": SCENARIOS, "controllers": CONTROLLERS, "models_available": bundle is not None,
        "horizons_min": cfg["forecasting"]["horizons_min"],
        "research_workload": "Research profile expands trajectories to 2 days, 10 replications, 20 GRU epochs, and a larger search budget; expect a long CPU run.",
    })


@api.post("/api/simulations")
def create_simulation():
    data = request.get_json(silent=True) or {}
    seed = bounded_int(data["seed"], "seed", 0, 2**32 - 1) if "seed" in data else None
    result = _services()["simulations"].create(
        str(data.get("scenario", "normal")), str(data.get("controller", "reactive")),
        finite_number(data.get("speed", 1.0), "speed"), seed,
    )
    return jsonify(result), 201


@api.get("/api/simulations/<simulation_id>")
def get_simulation(simulation_id: str):
    return jsonify(_services()["simulations"].describe(valid_id(simulation_id)))


@api.post("/api/simulations/<simulation_id>/control")
def control_simulation(simulation_id: str):
    data = request.get_json(silent=True) or {}
    speed = finite_number(data["speed"], "speed") if "speed" in data else None
    return jsonify(_services()["simulations"].control(valid_id(simulation_id), str(data.get("command", "")), speed))


@api.get("/api/simulations/<simulation_id>/history")
def simulation_history(simulation_id: str):
    maximum = int(_services()["config"]["service"]["max_history"])
    limit = bounded_int(request.args.get("limit", 240), "limit", 1, maximum)
    return jsonify({"history": _services()["simulations"].history(valid_id(simulation_id), limit)})


@api.get("/api/simulations/<simulation_id>/forecast")
def simulation_forecast(simulation_id: str):
    cfg = _services()["config"]
    rack = bounded_int(request.args.get("rack_id", 0), "rack_id", 0, int(cfg["topology"]["racks"]) - 1)
    horizon = bounded_int(request.args.get("horizon_min", 30), "horizon_min", 1, int(cfg["service"]["max_horizon_min"]))
    return jsonify(_services()["simulations"].forecast(valid_id(simulation_id), rack, horizon))


@api.get("/api/simulations/<simulation_id>/decisions")
def simulation_decisions(simulation_id: str):
    return jsonify({"decisions": _services()["simulations"].decisions(valid_id(simulation_id))})


@api.post("/api/jobs")
def create_job():
    data = request.get_json(silent=True) or {}
    profile = str(data.get("profile", "smoke"))
    if profile == "research" and data.get("confirm_research") is not True:
        raise ValueError("research profile requires explicit confirmation because it can take hours")
    return jsonify(_services()["jobs"].create(profile)), 202


@api.get("/api/jobs")
def list_jobs():
    return jsonify({"jobs": _services()["jobs"].list()})


@api.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    return jsonify(_services()["jobs"].get(valid_id(job_id)))


@api.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    return jsonify(_services()["jobs"].cancel(valid_id(job_id)))


def _run_directory(run_id: str) -> Path:
    valid_id(run_id)
    root = project_path(_services()["config"], "reports_dir").resolve()
    path = (root / run_id).resolve()
    if path.parent != root or not path.is_dir():
        raise KeyError(run_id)
    return path


@api.get("/api/runs")
def runs():
    root = project_path(_services()["config"], "reports_dir")
    result = []
    if root.is_dir():
        for path in sorted(root.iterdir(), reverse=True):
            manifest = path / "manifest.json"
            if path.is_dir() and manifest.is_file():
                import json
                data = json.loads(manifest.read_text(encoding="utf-8"))
                result.append({"run_id": path.name, "status": data.get("status"), "profile": data.get("profile"), "created_utc": data.get("created_utc"), "wall_time_s": data.get("wall_time_s"), "artifacts": artifact_catalog(path)})
    return jsonify({"runs": result})


@api.get("/api/runs/<run_id>")
def run_details(run_id: str):
    import json
    path = _run_directory(run_id)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8")) if (path / "summary.json").is_file() else None
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return jsonify({"run_id": run_id, "manifest": manifest, "summary": summary, "artifacts": artifact_catalog(path)})


@api.get("/api/runs/<run_id>/artifacts/<path:artifact_id>")
def download_artifact(run_id: str, artifact_id: str):
    from thermotwin.services.artifacts import resolve_artifact
    path = resolve_artifact(_run_directory(run_id), artifact_id)
    return send_file(path, as_attachment=path.suffix.lower() not in {".html", ".png", ".svg"}, download_name=path.name)


@api.app_errorhandler(ValueError)
def value_error(error):
    return jsonify({"error": "invalid_request", "message": str(error)}), 400


@api.app_errorhandler(KeyError)
def key_error(error):
    return jsonify({"error": "not_found", "message": "resource not found"}), 404


@api.app_errorhandler(HTTPException)
def http_error(error):
    return jsonify({"error": error.name.lower().replace(" ", "_"), "message": error.description}), error.code


@api.app_errorhandler(Exception)
def unexpected_error(error):
    current_app.logger.exception("Unhandled ThermoTwin error")
    return jsonify({"error": "internal_error", "message": "An unexpected server error occurred"}), 500
