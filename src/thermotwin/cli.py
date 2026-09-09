"""Command-line interface for diagnostics, preparation, experiments, simulation, reports, and serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import flask
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from thermotwin.config import load_config, project_path, validate_config
from thermotwin.data.features import build_supervised
from thermotwin.data.generate import generate_dataset, load_dataset
from thermotwin.experiments.report import generate_report
from thermotwin.experiments.runner import run_experiments
from thermotwin.forecasting.evaluate import regression_rows
from thermotwin.forecasting.registry import load_bundle
from thermotwin.forecasting.train import train_all
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.workload import scenario_inputs
from thermotwin.web import create_app
from thermotwin.workflows import load_prepared, prepare_project, prepared_paths


def _config(args) -> dict:
    config = load_config(getattr(args, "config", None), getattr(args, "profile", None))
    if getattr(args, "seed", None) is not None:
        config["project"]["seed"] = int(args.seed)
        validate_config(config)
    return config


def _progress(message: str) -> None:
    print(f"[ThermoTwin] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermotwin", description="ThermoTwin simulated thermal digital-twin research platform")
    sub = parser.add_subparsers(dest="command", required=True)
    def common(name, help_text):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--profile", choices=["default", "smoke", "research"], default="smoke")
        item.add_argument("--config", type=Path)
        return item
    common("doctor", "check interpreter, packages, configuration, and prepared artifacts")
    prepare = common("prepare", "generate data, train every model family, run six experiments, and report")
    prepare.add_argument("--force", action="store_true", help="rebuild matching cached artifacts")
    prepare.add_argument("--seed", type=int)
    generate = common("generate", "generate reproducible action-varied synthetic data")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--seed", type=int)
    train = common("train", "train and persist all required forecast model families")
    train.add_argument("--input", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--seed", type=int)
    evaluate = common("evaluate", "evaluate prepared models on the leakage-safe test block")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--seed", type=int)
    simulate = common("simulate", "run a bounded closed-loop simulation in the terminal")
    simulate.add_argument("--scenario", choices=["normal", "burst", "skewed", "hot_ambient", "cooling_degradation"], default="normal")
    simulate.add_argument("--steps", type=int, default=30)
    simulate.add_argument("--seed", type=int)
    experiments = common("experiments", "run one or all experiments using prepared artifacts")
    experiments.add_argument("--experiment", type=int, choices=range(1, 7), action="append")
    experiments.add_argument("--output", type=Path)
    experiments.add_argument("--force", action="store_true")
    experiments.add_argument("--seed", type=int)
    report = sub.add_parser("report", help="regenerate an offline HTML report from a run summary")
    report.add_argument("run_id")
    serve = common("serve", "serve the local dashboard and same-origin API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        config = _config(args)
        bundle, _ = load_prepared(config)
        print(json.dumps({
            "status": "ok", "python": platform.python_version(), "executable": sys.executable,
            "packages": {"flask": flask.__version__ if hasattr(flask, "__version__") else "installed", "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__, "torch": torch.__version__},
            "profile": config["project"]["profile"], "models_available": bundle is not None,
        }, indent=2))
        return 0
    if args.command == "prepare":
        run_dir, record = prepare_project(args.profile, args.force, _progress, config_path=args.config, seed=args.seed)
        print(json.dumps({"run_dir": str(run_dir), **record}, indent=2))
        return 0
    config = _config(args) if args.command != "report" else load_config(profile="smoke")
    dataset, models, _ = prepared_paths(config)
    if args.command == "generate":
        path = args.output or dataset
        frame = generate_dataset(config, path, progress=_progress)
        print(f"Wrote {len(frame):,} rows to {path}")
        return 0
    if args.command == "train":
        input_path, output_path = args.input or dataset, args.output or models
        _, metadata = train_all(config, input_path, output_path, _progress)
        print(json.dumps(metadata, indent=2, default=str))
        return 0
    if args.command == "evaluate":
        bundle, _ = load_bundle(models)
        parts, _ = build_supervised(load_dataset(dataset), config)
        test = parts["test"]
        predicted = test.nominal + bundle["hybrid"].predict(test.X)
        rows = regression_rows("hybrid", test.y, predicted, config["forecasting"]["horizons_min"])
        output = args.output or models.parent / "evaluation.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        return 0
    if args.command == "simulate":
        if not 1 <= args.steps <= 10000:
            raise SystemExit("--steps must be between 1 and 10000")
        plant = DataCenterPlant(config, args.seed)
        rng = np.random.default_rng((args.seed or config["project"]["seed"]) ^ 0xC0FFEE)
        for step in range(args.steps):
            plant.advance(*scenario_inputs(step, plant.n_racks, args.scenario, rng))
        snapshot = plant.history[-1]
        print(json.dumps({"scenario": args.scenario, "steps": args.steps, "simulated_minutes": snapshot.time_s / 60, "peak_temperature_c": float(max(plant.hidden.temperatures_c)), "total_power_kw": float((sum(snapshot.it_power_w) + snapshot.cooling_power_w + snapshot.fan_power_w) / 1000)}, indent=2))
        return 0
    if args.command == "experiments":
        bundle, metadata = load_bundle(models)
        run_dir, summary = run_experiments(config, dataset, bundle, metadata, args.output, args.experiment, resume=not args.force, progress=_progress)
        print(json.dumps({"run_dir": str(run_dir), "status": summary["status"]}, indent=2))
        return 0
    if args.command == "report":
        root = project_path(config, "reports_dir")
        run_dir = root / args.run_id
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"Run summary not found: {summary_path}")
        path = generate_report(run_dir, json.loads(summary_path.read_text(encoding="utf-8")))
        print(path)
        return 0
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost"}:
            raise SystemExit("For safety, ThermoTwin binds to 127.0.0.1/localhost only")
        if not 1 <= args.port <= 65535:
            raise SystemExit("--port must be between 1 and 65535")
        app = create_app(args.config, args.profile)
        print(f"ThermoTwin dashboard: http://127.0.0.1:{args.port}")
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
        return 0
    return 2
