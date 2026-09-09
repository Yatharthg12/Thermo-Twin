"""Common resumable runner implementing all six required experiments and artifacts."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd

from thermotwin.config import config_fingerprint, project_path
from thermotwin.control.policies import make_policy
from thermotwin.data.features import build_supervised
from thermotwin.data.generate import load_dataset
from thermotwin.experiments.ablation import ABLATION_ARMS
from thermotwin.experiments.plots import bar_plot, line_plot
from thermotwin.experiments.report import generate_report
from thermotwin.experiments.scenarios import run_closed_loop, topology_config
from thermotwin.experiments.statistics import paired_bootstrap_interval
from thermotwin.forecasting.baselines import persistence_predict
from thermotwin.forecasting.evaluate import hotspot_metrics, regression_rows, timed_predict, uncertainty_rows
from thermotwin.physics.cooling_model import advance_actuator, capacity_limited_conductance
from thermotwin.physics.thermal_model import chain_coupling, rk4_step, thermal_derivative
from thermotwin.physics.twin import DigitalTwin, LearnedOnlyTwin
from thermotwin.services.artifacts import artifact_catalog, atomic_json, create_manifest, source_fingerprint


EXPERIMENT_NAMES = {
    1: "physics_numerics", 2: "forecast_benchmark", 3: "horizon_sensitivity",
    4: "closed_loop_control", 5: "ablation", 6: "robustness_scale",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _experiment_one(config: dict, run_dir: Path) -> dict[str, Any]:
    capacitance, conductance, inlet, initial, power = 900000.0, 620.0, 20.0, 30.0, 0.0
    duration, step = 1800.0, 10.0
    times = np.arange(0, duration + step, step)
    state = np.array([initial])
    values = [initial]
    derivative = lambda temp: thermal_derivative(temp, np.array([power]), np.zeros(1), np.array([inlet]), np.array([conductance]), np.zeros((1, 1)), np.array([capacitance]))
    for _ in times[1:]:
        state = rk4_step(state, step, derivative)
        values.append(float(state[0]))
    analytic = inlet + (initial - inlet) * np.exp(-conductance * times / capacitance)
    analytic_error = float(np.max(np.abs(np.asarray(values) - analytic)))
    equilibrium_derivative = float(abs(thermal_derivative(np.array([20.0]), np.zeros(1), np.zeros(1), np.array([20.0]), np.array([620.0]), np.zeros((1, 1)), np.array([capacitance]))[0]))
    heat_derivative = float(thermal_derivative(np.array([20.0]), np.array([5000.0]), np.zeros(1), np.array([20.0]), np.array([620.0]), np.zeros((1, 1)), np.array([capacitance]))[0])
    zones = np.array([0, 0, 0])
    coupling = chain_coupling(3, zones, 42.0)
    coupling_balance = float(abs(np.sum(thermal_derivative(np.array([20., 30., 25.]), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), coupling, np.ones(3)))))
    actuator = advance_actuator(np.array([20.]), np.array([15.]), 180., 1 / 60, 60.)
    _, extracted = capacity_limited_conductance(np.array([60., 60.]), np.array([20., 20.]), 1000., np.array([1.]), np.array([0, 0]), 10000., 1.)
    low_flow, _ = capacity_limited_conductance(np.array([30.]), np.array([20.]), 620., np.array([.6]), np.array([0]), 42000., .72)
    high_flow, _ = capacity_limited_conductance(np.array([30.]), np.array([20.]), 620., np.array([1.2]), np.array([0]), 42000., .72)
    def integrate(dt: float) -> float:
        state = np.array([initial])
        for _ in range(int(duration / dt)):
            state = rk4_step(state, dt, derivative)
        return float(state[0])
    ref = float(analytic[-1])
    err60, err15 = abs(integrate(60) - ref), abs(integrate(15) - ref)
    rows = pd.DataFrame([
        {"check": "analytic decoupled cooling", "measured": analytic_error, "limit": 1e-6, "passed": analytic_error < 1e-6, "units": "°C"},
        {"check": "zero-input equilibrium", "measured": equilibrium_derivative, "limit": 1e-12, "passed": equilibrium_derivative < 1e-12, "units": "K/s"},
        {"check": "positive heat response", "measured": heat_derivative, "limit": 0.0, "passed": heat_derivative > 0, "units": "K/s"},
        {"check": "symmetric coupling conservation", "measured": coupling_balance, "limit": 1e-12, "passed": coupling_balance < 1e-12, "units": "W-equivalent"},
        {"check": "actuator finite lag", "measured": float(actuator[0]), "limit": 19.0, "passed": 15 < actuator[0] < 20, "units": "°C"},
        {"check": "capacity saturation", "measured": float(extracted[0]), "limit": 10000.0, "passed": extracted[0] <= 10000.0001, "units": "W"},
        {"check": "airflow increases conductance", "measured": float(high_flow[0] - low_flow[0]), "limit": 0.0, "passed": high_flow[0] > low_flow[0], "units": "W/K difference"},
        {"check": "timestep refinement", "measured": err15, "limit": err60 + 1e-12, "passed": err15 <= err60 + 1e-12, "units": "°C error"},
    ])
    _write_csv(rows, run_dir / "experiment_1_physics.csv")
    plot_frame = pd.DataFrame({"time_min": np.tile(times / 60, 2), "temperature_c": np.r_[values, analytic], "series": ["RK4 (10 s)"] * len(times) + ["analytic"] * len(times)})
    line_plot(plot_frame, "time_min", "temperature_c", "series", "Analytic/reference thermal verification", "Simulated time (min)", "Effective rack temperature (°C)", run_dir / "physics_verification")
    return {"status": "complete", "checks_passed": int(rows["passed"].sum()), "checks_total": len(rows), "raw": rows.to_dict("records")}


def _forecast_predictions(bundle: dict, part: Any) -> dict[str, tuple[np.ndarray, float]]:
    predictions = {
        "persistence": timed_predict(persistence_predict, part.X, part.y.shape[1]),
        "physics": timed_predict(lambda x: x.copy(), part.nominal),
    }
    for name, model in bundle["classical"].items():
        predictions[name] = timed_predict(model.predict, part.X)
    predictions["recurrent"] = timed_predict(bundle["recurrent"].predict, part.X_sequence)
    predictions["hybrid"] = timed_predict(lambda: part.nominal + bundle["hybrid"].predict(part.X))
    return predictions


def _experiment_two(config: dict, run_dir: Path, bundle: dict, metadata: dict, parts: dict) -> dict[str, Any]:
    test = parts["test"]
    predictions = _forecast_predictions(bundle, test)
    rows = []
    rack_rows = []
    scenario_rows = []
    horizons = config["forecasting"]["horizons_min"]
    for name, (predicted, elapsed) in predictions.items():
        for row in regression_rows(name, test.y, predicted, horizons, elapsed):
            row["training_s"] = float(metadata.get("training_times_s", {}).get(name, 0.0))
            row["scenario_count"] = int(len(np.unique(test.run_ids)))
            rows.append(row)
        for rack in np.unique(test.rack_ids):
            mask = test.rack_ids == rack
            for row in regression_rows(name, test.y[mask], predicted[mask], horizons, elapsed):
                row["rack_id"] = int(rack)
                rack_rows.append(row)
        for scenario in np.unique(test.scenario_ids):
            mask = test.scenario_ids == scenario
            for row in regression_rows(name, test.y[mask], predicted[mask], horizons, elapsed):
                row["scenario"] = str(scenario)
                row["heldout_distribution_shift"] = scenario in set(config["forecasting"].get("heldout_scenarios", []))
                scenario_rows.append(row)
    frame = pd.DataFrame(rows)
    _write_csv(frame, run_dir / "experiment_2_forecasts.csv")
    _write_csv(pd.DataFrame(rack_rows), run_dir / "experiment_2_by_rack.csv")
    _write_csv(pd.DataFrame(scenario_rows), run_dir / "experiment_2_by_scenario.csv")
    plot = frame.groupby("model", as_index=False)["mae_c"].mean().sort_values("mae_c")
    bar_plot(plot, "model", "mae_c", "Held-out forecast benchmark", "Mean MAE across horizons (°C)", run_dir / "forecast_benchmark")
    return {"status": "complete", "validation_selected_benchmark": min(metadata["validation_mae_c"], key=metadata["validation_mae_c"].get), "operational_model": "hybrid", "rows": len(frame)}


def _experiment_three(config: dict, run_dir: Path, bundle: dict, parts: dict) -> dict[str, Any]:
    test = parts["test"]
    predicted = test.nominal + bundle["hybrid"].predict(test.X)
    lower, upper = bundle["interval"].interval(predicted)
    reg = regression_rows("hybrid", test.y, predicted, config["forecasting"]["horizons_min"])
    uncertainty = uncertainty_rows(test.y, lower, upper, config["forecasting"]["horizons_min"], bundle["interval"].level)
    rows = []
    for metrics, interval in zip(reg, uncertainty):
        rows.append({**metrics, **{key: value for key, value in interval.items() if key != "horizon_min"}})
    frame = pd.DataFrame(rows)
    _write_csv(frame, run_dir / "experiment_3_horizons.csv")
    line_plot(frame, "horizon_min", "mae_c", "model", "Forecast error versus horizon", "Forecast horizon (min)", "MAE (°C)", run_dir / "horizon_error")
    widths = upper - lower
    probabilities = bundle["hotspot"].predict_proba(predicted, widths)
    event_rows = []
    for index, horizon in enumerate(config["forecasting"]["horizons_min"]):
        event_rows.append({"horizon_min": horizon, **hotspot_metrics(test.hotspot_events[:, index], probabilities[:, index])})
    event_frame = pd.DataFrame(event_rows)
    _write_csv(event_frame, run_dir / "experiment_3_hotspot.csv")
    event = event_rows[-1]
    atomic_json(run_dir / "experiment_3_hotspot.json", {"rows": event_rows})
    return {"status": "complete", "uncertainty_rows": len(frame), "hotspot_event_metrics_60_min": event,
            "event_definition": "Any true effective-rack threshold exceedance at any one-minute sample within each complete future window"}


def _policy_factory(name: str, config: dict, bundle: dict):
    if name == "learned_only":
        twin = LearnedOnlyTwin(config, bundle["learned_transition"])
    else:
        twin = DigitalTwin(config, bundle["action_residual"])
    return lambda: make_policy(name, config, twin)


def _experiment_four(config: dict, run_dir: Path, bundle: dict, cancel: Callable[[], bool] | None, progress: Callable[[str], None] | None) -> dict[str, Any]:
    rows, raw_decisions = [], {}
    base_seed = int(config["project"]["seed"]) + 4000
    steps = int(config["experiments"]["control_steps"])
    controllers = ["fixed", "reactive", "physics_mpc", "hybrid_mpc"]
    for replication in range(int(config["experiments"]["repetitions"])):
        seed = base_seed + replication
        for scenario in config["experiments"]["scenarios"]:
            for controller in controllers:
                if progress:
                    progress(f"Experiment 4: replication {replication + 1} / {scenario} / {controller}")
                metrics, decisions, _ = run_closed_loop(config, scenario, controller, _policy_factory(controller, config, bundle), seed, steps, cancel)
                metrics["replication"] = replication
                rows.append(metrics)
                raw_decisions[f"rep{replication}:{scenario}:{controller}"] = decisions
    frame = pd.DataFrame(rows)
    _write_csv(frame, run_dir / "experiment_4_control.csv")
    atomic_json(run_dir / "experiment_4_decisions.json", raw_decisions)
    aggregate = frame.groupby("controller", as_index=False).agg(total_energy_kwh=("total_energy_kwh", "mean"), peak_temperature_c=("peak_temperature_c", "max"), service_fraction=("service_fraction", "mean"))
    bar_plot(aggregate, "controller", "total_energy_kwh", "Closed-loop modeled energy", "Mean modeled energy (kWh)", run_dir / "control_energy")
    paired = frame.groupby(["replication", "controller"], as_index=False)["total_energy_kwh"].mean()
    fixed = paired[paired.controller == "fixed"].sort_values("replication")["total_energy_kwh"].to_numpy()
    hybrid = paired[paired.controller == "hybrid_mpc"].sort_values("replication")["total_energy_kwh"].to_numpy()
    evidence = paired_bootstrap_interval(fixed, hybrid, base_seed)
    return {"status": "complete", "rows": len(frame), "paired_hybrid_minus_fixed_energy_kwh": evidence}


def _experiment_five(config: dict, run_dir: Path, bundle: dict, cancel: Callable[[], bool] | None, progress: Callable[[str], None] | None) -> dict[str, Any]:
    rows, decisions_by_arm = [], {}
    seed = int(config["project"]["seed"]) + 5000
    for arm, removed in ABLATION_ARMS.items():
        if progress:
            progress(f"Experiment 5: {arm}")
        metrics, decisions, _ = run_closed_loop(config, "burst", arm, _policy_factory(arm, config, bundle), seed, int(config["experiments"]["control_steps"]), cancel)
        metrics["arm_description"] = removed
        rows.append(metrics)
        decisions_by_arm[arm] = decisions
    frame = pd.DataFrame(rows)
    _write_csv(frame, run_dir / "experiment_5_ablation.csv")
    atomic_json(run_dir / "experiment_5_decisions.json", decisions_by_arm)
    bar_plot(frame, "controller", "total_energy_kwh", "Controller component ablation", "Modeled energy (kWh)", run_dir / "ablation_energy")
    return {"status": "complete", "rows": len(frame), "interpretation": "Arms remove one controller component; interactions mean differences are not independent causal effects."}


def _experiment_six(config: dict, run_dir: Path, bundle: dict, cancel: Callable[[], bool] | None, progress: Callable[[str], None] | None) -> dict[str, Any]:
    rows = []
    seed = int(config["project"]["seed"]) + 6000
    settings = [
        ("mismatch", "0.00", {"physics": {"mismatch_fraction": 0.0}}),
        ("mismatch", "0.20", {"physics": {"mismatch_fraction": 0.2}}),
        ("sensor_quality", "low_noise", {"sensors": {"noise_std_c": 0.05, "missing_probability": 0.0}}),
        ("sensor_quality", "degraded", {"sensors": {"noise_std_c": 0.5, "missing_probability": 0.15}}),
        ("cooling_efficiency", "nominal", {}),
        ("cooling_efficiency", "degraded", {}),
        ("objective_energy_weight", "0.5", {"control": {"weights": {"energy": 0.5}}}),
        ("objective_energy_weight", "2.0", {"control": {"weights": {"energy": 2.0}}}),
    ]
    def deep_update(target, update):
        for key, value in update.items():
            if isinstance(value, dict):
                deep_update(target[key], value)
            else:
                target[key] = value
    for factor, level, override in settings:
        cfg = copy.deepcopy(config)
        deep_update(cfg, override)
        scenario = "cooling_degradation" if factor == "cooling_efficiency" and level == "degraded" else "normal"
        tick = time.perf_counter()
        metrics, _, _ = run_closed_loop(cfg, scenario, "hybrid_mpc", _policy_factory("hybrid_mpc", cfg, bundle), seed, min(12, int(cfg["experiments"]["control_steps"])), cancel)
        metrics.update({"factor": factor, "level": level, "racks": cfg["topology"]["racks"], "runtime_s": time.perf_counter() - tick, "memory_mb": None, "memory_reason": "portable process memory measurement unavailable"})
        rows.append(metrics)
    for racks in config["experiments"]["scale_racks"]:
        if progress:
            progress(f"Experiment 6: topology scale {racks} racks")
        cfg = topology_config(config, int(racks))
        tick = time.perf_counter()
        metrics, _, _ = run_closed_loop(cfg, "normal", "physics_mpc", _policy_factory("physics_mpc", cfg, bundle), seed, min(10, int(cfg["experiments"]["control_steps"])), cancel)
        metrics.update({"factor": "topology_scale", "level": str(racks), "racks": racks, "runtime_s": time.perf_counter() - tick, "memory_mb": None, "memory_reason": "portable process memory measurement unavailable"})
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    _write_csv(frame, run_dir / "experiment_6_robustness.csv")
    scale = frame[frame.factor == "topology_scale"].copy()
    line_plot(scale, "racks", "runtime_s", "controller", "Bounded topology scaling runtime", "Rack count", "Measured wall time (s)", run_dir / "scale_runtime")
    return {"status": "complete", "rows": len(frame), "compute_budget": "12 control steps per robustness setting and 10 per topology in smoke/default runner"}


def run_experiments(
    config: dict[str, Any], dataset_path: str | Path, bundle: dict[str, Any], model_metadata: dict[str, Any],
    output_root: str | Path | None = None, selected: list[int] | None = None, resume: bool = True,
    progress: Callable[[str], None] | None = None, cancel: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run selected or all experiments, persist structured evidence, and produce an offline report."""
    selected = selected or list(EXPERIMENT_NAMES)
    run_identity = "-".join([
        source_fingerprint(), config_fingerprint(config), str(model_metadata.get("data_fingerprint", "no-data")),
        str(model_metadata.get("model_id", "no-model")),
    ])
    fingerprint = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()
    root = Path(output_root) if output_root else project_path(config, "reports_dir")
    root.mkdir(parents=True, exist_ok=True)
    matching = sorted(root.glob(f"*-{config['project']['profile']}-{fingerprint[:8]}"))
    if resume and matching and (matching[-1] / "summary.json").is_file():
        existing = json.loads((matching[-1] / "summary.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and set(map(int, existing.get("experiments", {}))) >= set(selected):
            return matching[-1], existing
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config['project']['profile']}-{fingerprint[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_manifest(run_id, config)
    seed_path = Path(dataset_path).with_name(f"{Path(dataset_path).stem}_seeds.json")
    if seed_path.is_file():
        manifest["seeds"] = json.loads(seed_path.read_text(encoding="utf-8"))
    manifest.update({
        "run_fingerprint": fingerprint,
        "data_path": str(Path(dataset_path).name),
        "data_fingerprint": model_metadata.get("data_fingerprint"),
        "model_id": model_metadata.get("model_id"),
        "model_config_fingerprint": model_metadata.get("config_fingerprint"),
        "split_ranges": model_metadata.get("split_ranges", []),
        "selected_experiments": selected,
    })
    atomic_json(run_dir / "manifest.json", manifest)
    frame = load_dataset(dataset_path)
    parts, _ = build_supervised(frame, config)
    results: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    functions = {
        1: lambda: _experiment_one(config, run_dir),
        2: lambda: _experiment_two(config, run_dir, bundle, model_metadata, parts),
        3: lambda: _experiment_three(config, run_dir, bundle, parts),
        4: lambda: _experiment_four(config, run_dir, bundle, cancel, progress),
        5: lambda: _experiment_five(config, run_dir, bundle, cancel, progress),
        6: lambda: _experiment_six(config, run_dir, bundle, cancel, progress),
    }
    try:
        for number in selected:
            if cancel and cancel():
                raise InterruptedError("experiment cancelled")
            if progress:
                progress(f"Starting experiment {number}: {EXPERIMENT_NAMES[number]}")
            tick = time.perf_counter()
            result = functions[number]()
            result.update({"name": EXPERIMENT_NAMES[number], "wall_time_s": time.perf_counter() - tick, "seed": config["project"]["seed"], "config_fingerprint": config_fingerprint(config), "data_fingerprint": model_metadata.get("data_fingerprint"), "model_id": model_metadata.get("model_id")})
            results[number] = result
            atomic_json(run_dir / f"experiment_{number}_config.json", config)
            atomic_json(run_dir / f"experiment_{number}_status.json", result)
        status = "complete"
    except InterruptedError:
        status = "cancelled"
    except Exception as error:
        status = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest.update({"status": status, "wall_time_s": time.perf_counter() - started, "completed_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(run_dir / "manifest.json", manifest)
    summary = {
        "run_id": run_id, "profile": config["project"]["profile"], "status": status,
        "experiments": {str(key): value for key, value in results.items()},
        "provenance": {"source_fingerprint": source_fingerprint(), "config_fingerprint": config_fingerprint(config), "data_fingerprint": model_metadata.get("data_fingerprint"), "model_id": model_metadata.get("model_id"), "model_created_utc": model_metadata.get("created_utc"), "run_wall_time_s": manifest["wall_time_s"]},
    }
    atomic_json(run_dir / "summary.json", summary)
    if status == "complete":
        generate_report(run_dir, summary)
    # Writing a self-catalogued byte length would change that length. The
    # manifest remains downloadable through the scoped artifact resolver.
    manifest["artifacts"] = [item for item in artifact_catalog(run_dir) if item["artifact_id"] != "manifest.json"]
    atomic_json(run_dir / "manifest.json", manifest)
    return run_dir, summary
