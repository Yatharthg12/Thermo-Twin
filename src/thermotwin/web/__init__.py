"""Flask application factory for the same-origin offline dashboard and JSON API."""

from __future__ import annotations

import atexit
from flask import Flask

from thermotwin.config import load_config
from thermotwin.services.artifacts import source_fingerprint
from thermotwin.services.job_manager import JobManager
from thermotwin.services.simulation_service import SimulationService
from thermotwin.workflows import load_prepared, prepare_project


def create_app(config_path: str | None = None, profile: str = "smoke", testing: bool = False) -> Flask:
    """Create an isolated Flask app; no training or simulation advancement occurs at construction."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(TESTING=testing, JSON_SORT_KEYS=False, MAX_CONTENT_LENGTH=64 * 1024)
    config = load_config(config_path, profile)
    bundle, model_metadata = load_prepared(config)
    simulation_service = SimulationService(config, bundle)
    def job_runner(job_profile, progress, cancel):
        run_dir, _ = prepare_project(job_profile, False, progress, cancel)
        return run_dir.name
    job_manager = JobManager(config, job_runner)
    app.extensions["thermotwin"] = {
        "config": config, "bundle": bundle, "model_metadata": model_metadata,
        "simulations": simulation_service, "jobs": job_manager,
        "source_fingerprint": source_fingerprint(),
    }
    from thermotwin.web.routes import api
    app.register_blueprint(api)
    atexit.register(simulation_service.shutdown)
    atexit.register(job_manager.shutdown)
    return app
