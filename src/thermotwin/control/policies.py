"""Fixed, reactive, physics-only, hybrid, learned-only, rule, and no-migration policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
import numpy as np

from thermotwin.control.actions import generate_candidates
from thermotwin.control.optimizer import optimize
from thermotwin.physics.twin import DigitalTwin
from thermotwin.types import Action, DecisionRecord, Snapshot


class BasePolicy(ABC):
    """Minimal stateful policy interface used identically by simulation and experiments."""

    name = "base"

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> DecisionRecord:
        """Choose one feasible action using policy-visible state only."""


class FixedPolicy(BasePolicy):
    """Hold configured settings and never migrate."""

    name = "fixed"

    def decide(self, snapshot: Snapshot) -> DecisionRecord:
        action = Action.noop(len(snapshot.supply_actual_c))
        return DecisionRecord(snapshot.time_s, self.name, action, [], False, "Fixed cooling holds configured commands", 0.0)


class ReactivePolicy(BasePolicy):
    """Representative max-temperature hysteresis baseline using current estimates only."""

    name = "reactive"

    def __init__(self, config: dict):
        self.config = config
        self.engaged = False

    def decide(self, snapshot: Snapshot) -> DecisionRecord:
        started = time.perf_counter()
        peak = float(np.max(snapshot.estimated_temperatures_c))
        if peak > self.config["thresholds"]["hotspot_c"] - 1.5:
            self.engaged = True
        elif peak < self.config["thresholds"]["hotspot_c"] - 3.5:
            self.engaged = False
        if self.engaged:
            cooling = self.config["cooling"]
            supply = tuple(float(max(cooling["supply_min_c"] - value, -0.5)) for value in snapshot.supply_command_c)
            airflow = tuple(float(min(cooling["airflow_max"] - value, 0.08)) for value in snapshot.airflow_command)
            action = Action("reactive-cooling", supply, airflow, None, "Hysteresis cooling response")
        else:
            action = Action.noop(len(snapshot.supply_actual_c))
        return DecisionRecord(snapshot.time_s, self.name, action, [], False, "Current-temperature hysteresis (on at hotspot−1.5 °C, off at hotspot−3.5 °C)", (time.perf_counter() - started) * 1000)


class PredictivePolicy(BasePolicy):
    """Bounded MPC-style candidate search through nominal or learned-residual twin rollouts."""

    def __init__(self, config: dict, twin: DigitalTwin, use_residual: bool, allow_migration: bool = True, name: str | None = None):
        self.config, self.twin, self.use_residual, self.allow_migration = config, twin, use_residual, allow_migration
        self.name = name or ("hybrid_mpc" if use_residual else "physics_mpc")

    def decide(self, snapshot: Snapshot) -> DecisionRecord:
        candidates = generate_candidates(snapshot, self.config, self.allow_migration)
        return optimize(snapshot, candidates, self.twin, self.config, self.name, self.use_residual)


class ForecastRulePolicy(ReactivePolicy):
    """Deterministic forecast-triggered action without candidate optimization."""

    name = "no_optimization"

    def decide(self, snapshot: Snapshot) -> DecisionRecord:
        projected = float(np.max(snapshot.estimated_temperatures_c + 0.8 * (snapshot.utilization - 0.5)))
        self.engaged = projected > self.config["thresholds"]["hotspot_c"] - 2.0
        result = super().decide(snapshot)
        result.controller = self.name
        result.explanation = "Deterministic causal one-step forecast rule; no optimization"
        return result


def make_policy(name: str, config: dict, twin: DigitalTwin) -> BasePolicy:
    """Create a controller by stable API/experiment identifier."""
    if name == "fixed":
        return FixedPolicy()
    if name == "reactive":
        return ReactivePolicy(config)
    if name == "physics_mpc":
        return PredictivePolicy(config, twin, False)
    if name in {"hybrid_mpc", "full_hybrid"}:
        return PredictivePolicy(config, twin, True, name=name)
    if name == "no_migration":
        return PredictivePolicy(config, twin, True, allow_migration=False, name=name)
    if name == "learned_only":
        return PredictivePolicy(config, twin, True, name=name)
    if name == "no_optimization":
        return ForecastRulePolicy(config)
    raise ValueError(f"unknown controller: {name}")
