"""Counterfactual isolation, determinism, and action-dependence tests."""

import copy
import numpy as np

from thermotwin.config import load_config
from thermotwin.forecasting.hybrid import ResidualCorrector
from thermotwin.control.optimizer import optimize
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.twin import DigitalTwin
from thermotwin.types import Action


def test_counterfactual_does_not_mutate_plant_history_time_or_rng():
    config = load_config(profile="smoke")
    plant = DataCenterPlant(config, seed=23)
    snapshot = plant.snapshot()
    before = (plant.time_s, len(plant.history), copy.deepcopy(plant.process_rng.bit_generator.state), plant.hidden.temperatures_c.copy())
    twin = DigitalTwin(config)
    result = twin.simulate_action(snapshot, Action.noop(3), 300, {"utilization": snapshot.utilization, "ambient_c": 27.0})
    assert result.times_s[-1] == snapshot.time_s + 300
    assert plant.time_s == before[0] and len(plant.history) == before[1]
    assert plant.process_rng.bit_generator.state == before[2]
    assert np.array_equal(plant.hidden.temperatures_c, before[3])


def test_candidate_order_does_not_change_equivalent_rollouts():
    config = load_config(profile="smoke")
    snapshot = DataCenterPlant(config, seed=4).snapshot()
    twin = DigitalTwin(config)
    actions = [Action.noop(3), Action("cool", (-.5, 0, 0), (0, 0, 0), None, "cool")]
    forward = {a.action_id: twin.simulate_action(snapshot, a, 180, {"utilization": snapshot.utilization, "ambient_c": 27}).temperatures_c for a in actions}
    reverse = {a.action_id: twin.simulate_action(snapshot, a, 180, {"utilization": snapshot.utilization, "ambient_c": 27}).temperatures_c for a in reversed(actions)}
    for key in forward:
        assert np.allclose(forward[key], reverse[key])


def test_constructed_residual_changes_action_conditioned_rollout():
    config = load_config(profile="smoke")
    snapshot = DataCenterPlant(config, seed=4).snapshot()
    action_a = Action("a", (-.5, 0, 0), (0, 0, 0))
    action_b = Action("b", (.5, 0, 0), (0, 0, 0))
    twin = DigitalTwin(config, ResidualCorrector(fallback_gain=2.0))
    inputs = {"utilization": snapshot.utilization, "ambient_c": 27}
    a = twin.simulate_action(snapshot, action_a, 60, inputs, True)
    b = twin.simulate_action(snapshot, action_b, 60, inputs, True)
    assert not np.allclose(a.temperatures_c, b.temperatures_c)


def test_constructed_learned_residual_changes_candidate_ranking():
    config = load_config(profile="smoke")
    config["control"]["horizon_min"] = 1
    config["control"]["weights"] = {"energy": 0.0, "hotspot": 10.0, "hard_limit": 10.0, "imbalance": 0.0, "movement": 0.0, "migration": 0.0}
    plant = DataCenterPlant(config, seed=5)
    plant.estimated_c[:] = 34.8
    snapshot = plant.snapshot()
    cool = Action("cool", (-.5, 0, 0), (0, 0, 0))
    warm = Action("warm", (.5, 0, 0), (0, 0, 0))
    physics_choice = optimize(snapshot, [cool, warm], DigitalTwin(config), config, "physics", False).chosen_action.action_id
    def adversarial_residual(_snapshot, action):
        return np.full(12, 4.0 if action.action_id == "cool" else -4.0)
    learned_choice = optimize(snapshot, [cool, warm], DigitalTwin(config, adversarial_residual), config, "hybrid", True).chosen_action.action_id
    assert physics_choice != learned_choice
