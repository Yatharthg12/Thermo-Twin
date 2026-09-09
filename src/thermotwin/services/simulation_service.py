"""Authoritative per-simulation worker loop and bounded snapshot read interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any
import uuid

import numpy as np

from thermotwin.control.policies import make_policy
from thermotwin.experiments.metrics import summarize_snapshots
from thermotwin.forecasting.hybrid import ResidualCorrector
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.twin import DigitalTwin
from thermotwin.physics.workload import SCENARIOS, scenario_inputs
from thermotwin.types import Action


CONTROLLERS = ("fixed", "reactive", "physics_mpc", "hybrid_mpc")


@dataclass
class SimulationSession:
    """All mutable state owned by exactly one worker and serialized by one lock."""

    simulation_id: str
    config: dict[str, Any]
    scenario: str
    controller: str
    seed: int
    twin: DigitalTwin
    plant: DataCenterPlant = field(init=False)
    policy: Any = field(init=False)
    workload_rng: np.random.Generator = field(init=False)
    status: str = "initialized"
    speed: float = 1.0
    step_index: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    updated_monotonic: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    wake: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._reset_objects()

    def _reset_objects(self) -> None:
        self.plant = DataCenterPlant(self.config, self.seed)
        self.policy = make_policy(self.controller, self.config, self.twin)
        self.workload_rng = np.random.default_rng(self.seed ^ 0xC0FFEE)
        self.step_index, self.decisions, self.error = 0, [], None

    def step_once(self) -> None:
        offered, ambient, humidity = scenario_inputs(self.step_index, self.plant.n_racks, self.scenario, self.workload_rng)
        action = Action.noop(self.plant.n_zones)
        interval = int(self.config["control"]["decision_interval_steps"])
        if self.step_index % interval == 0:
            decision = self.policy.decide(self.plant.snapshot(ambient, humidity))
            action = decision.chosen_action
            self.decisions.append(decision.as_dict())
            self.decisions = self.decisions[-250:]
        accepted, reason = self.plant.apply_action(action)
        if not accepted:
            self.decisions.append({"time_s": self.plant.time_s, "controller": self.controller, "fallback": True, "explanation": f"Live action rejected: {reason}", "candidates": []})
        degradation = 0.72 if self.scenario == "cooling_degradation" and self.step_index >= 60 else None
        self.plant.advance(offered, ambient, humidity, degradation=degradation)
        max_history = int(self.config["service"]["max_history"])
        if len(self.plant.history) > max_history:
            self.plant.history = self.plant.history[-max_history:]
            self.plant.truth_history = self.plant.truth_history[-max_history:]
        self.step_index += 1
        self.updated_monotonic = time.monotonic()


class SimulationService:
    """Create, command, inspect, and terminate simulations without GET side effects."""

    def __init__(self, config: dict[str, Any], bundle: dict[str, Any] | None = None):
        self.config, self.bundle = config, bundle
        residual = bundle["action_residual"] if bundle else ResidualCorrector()
        self.twin = DigitalTwin(config, residual)
        self.sessions: dict[str, SimulationSession] = {}
        self.lock = threading.RLock()

    @property
    def trained_models_available(self) -> bool:
        return self.bundle is not None

    def create(self, scenario: str, controller: str, speed: float = 1.0, seed: int | None = None) -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario")
        if controller not in CONTROLLERS:
            raise ValueError("unknown controller")
        if controller == "hybrid_mpc" and not self.trained_models_available:
            raise ValueError("hybrid_mpc requires prepared models; run prepare --profile smoke")
        if not np.isfinite(speed) or not 0.25 <= speed <= 20:
            raise ValueError("speed must be finite and between 0.25 and 20")
        identifier = uuid.uuid4().hex[:12]
        session = SimulationSession(identifier, self.config, scenario, controller, int(seed or self.config["project"]["seed"]), self.twin, speed=float(speed))
        with self.lock:
            self.sessions[identifier] = session
        thread = threading.Thread(target=self._worker, args=(session,), name=f"thermotwin-sim-{identifier}", daemon=True)
        thread.start()
        return self.describe(identifier)

    def _worker(self, session: SimulationSession) -> None:
        while not session.stop.is_set():
            if session.status != "running":
                session.wake.wait(0.25)
                session.wake.clear()
                continue
            try:
                with session.lock:
                    if session.status == "running":
                        session.step_once()
            except Exception as error:
                with session.lock:
                    session.status, session.error = "failed", f"{type(error).__name__}: {error}"
            session.stop.wait(float(session.config["service"]["playback_interval_s"]) / session.speed)

    def get(self, simulation_id: str) -> SimulationSession:
        with self.lock:
            session = self.sessions.get(simulation_id)
        if session is None:
            raise KeyError(simulation_id)
        return session

    def control(self, simulation_id: str, command: str, speed: float | None = None) -> dict[str, Any]:
        session = self.get(simulation_id)
        if command not in {"play", "pause", "step", "reset", "speed"}:
            raise ValueError("unknown control command")
        with session.lock:
            if command == "play":
                if session.status == "failed":
                    raise ValueError("failed simulation must be reset")
                session.status = "running"
            elif command == "pause":
                session.status = "paused"
            elif command == "step":
                if session.status == "running":
                    raise ValueError("pause before single-step")
                session.step_once()
                session.status = "paused"
            elif command == "reset":
                session.status = "paused"
                session._reset_objects()
                session.updated_monotonic = time.monotonic()
            elif command == "speed":
                if speed is None or not np.isfinite(speed) or not 0.25 <= speed <= 20:
                    raise ValueError("speed must be finite and between 0.25 and 20")
                session.speed = float(speed)
            session.wake.set()
        return self.describe(simulation_id)

    def describe(self, simulation_id: str) -> dict[str, Any]:
        """Read state only; it never advances simulated time."""
        session = self.get(simulation_id)
        with session.lock:
            snapshot = session.plant.history[-1]
            metrics = summarize_snapshots(session.plant.history, session.config["thresholds"]["hotspot_c"], session.config["thresholds"]["hard_limit_c"])
            stale = session.status == "running" and time.monotonic() - session.updated_monotonic > 5
            return {
                "simulation_id": simulation_id, "status": "stale" if stale else session.status,
                "scenario": session.scenario, "controller": session.controller, "speed": session.speed,
                "step_index": session.step_index, "error": session.error, "snapshot": snapshot.as_dict(),
                "metrics": metrics, "models_available": self.trained_models_available,
            }

    def history(self, simulation_id: str, limit: int) -> list[dict[str, Any]]:
        session = self.get(simulation_id)
        maximum = int(session.config["service"]["max_history"])
        if not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")
        with session.lock:
            return [item.as_dict() for item in session.plant.history[-limit:]]

    def decisions(self, simulation_id: str) -> list[dict[str, Any]]:
        session = self.get(simulation_id)
        with session.lock:
            return list(session.decisions)

    def forecast(self, simulation_id: str, rack_id: int, horizon_min: int) -> dict[str, Any]:
        session = self.get(simulation_id)
        if rack_id not in range(session.plant.n_racks):
            raise ValueError("rack_id out of range")
        maximum = int(session.config["service"]["max_horizon_min"])
        if not 1 <= horizon_min <= maximum:
            raise ValueError(f"horizon_min must be between 1 and {maximum}")
        with session.lock:
            snapshot = session.plant.history[-1]
            history = session.plant.history[-90:]
        steps = int(horizon_min * 60 / session.config["control"]["rollout_step_s"])
        configured_horizons = np.asarray(session.config["forecasting"]["horizons_min"], dtype=int)
        projection_minutes = max(horizon_min, int(configured_horizons.max())) if self.bundle else horizon_min
        projection_steps = int(projection_minutes * 60 / session.config["control"]["rollout_step_s"])
        inputs = {"utilization": np.repeat(snapshot.utilization[None, :], projection_steps, axis=0), "ambient_c": np.full(projection_steps, snapshot.ambient_c)}
        noop = Action.noop(session.plant.n_zones)
        physics = self.twin.simulate_action(snapshot, noop, projection_minutes * 60, inputs, False)
        hybrid = self.twin.simulate_action(snapshot, noop, projection_minutes * 60, inputs, bool(self.bundle))
        future_minutes = np.arange(1, steps + 1)
        if self.bundle:
            radii = np.interp(future_minutes, configured_horizons, self.bundle["interval"].radii, left=self.bundle["interval"].radii[0], right=self.bundle["interval"].radii[-1])
        else:
            radii = np.full(steps, np.nan)
        predicted = hybrid.temperatures_c[:steps, rack_id]
        probability_horizon = None
        if self.bundle:
            direct = np.asarray([[hybrid.temperatures_c[horizon - 1, rack_id] for horizon in configured_horizons]])
            direct_widths = np.asarray([2.0 * self.bundle["interval"].radii])
            calibrated = self.bundle["hotspot"].predict_proba(direct, direct_widths)[0]
            probability_index = int(np.searchsorted(configured_horizons, horizon_min, side="left"))
            probability_index = min(probability_index, len(configured_horizons) - 1)
            hotspot_probability = float(calibrated[probability_index])
            probability_horizon = int(configured_horizons[probability_index])
        else:
            hotspot_probability = None
        return {
            "rack_id": rack_id, "forecast_origin_s": snapshot.time_s, "horizon_min": horizon_min,
            "history": [{"time_s": item.time_s, "temperature_c": float(item.estimated_temperatures_c[rack_id])} for item in history],
            "future": [{
                "time_s": float(physics.times_s[index]), "physics_c": float(physics.temperatures_c[index, rack_id]),
                "hybrid_c": float(predicted[index]),
                "lower_c": float(predicted[index] - radii[index]) if np.isfinite(radii[index]) else None,
                "upper_c": float(predicted[index] + radii[index]) if np.isfinite(radii[index]) else None,
            } for index in range(steps)],
            "interval_level": self.bundle["interval"].level if self.bundle else None,
            "hotspot_probability": hotspot_probability,
            "hotspot_probability_horizon_min": probability_horizon,
            "model_status": "trained hybrid residual" if self.bundle else "causal nominal fallback; run preparation for ML",
        }

    def shutdown(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session.stop.set()
            session.wake.set()
