"""Scientific invariants for thermal balance, integration, cooling, and energy units."""

import numpy as np

from thermotwin.config import load_config
from thermotwin.experiments.metrics import integrate_energy_kwh
from thermotwin.physics.cooling_model import advance_actuator, capacity_limited_conductance
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.thermal_model import chain_coupling, rk4_step, thermal_derivative


def test_decoupled_rk4_matches_analytic_and_refines():
    capacitance, conductance, inlet, initial = 900000.0, 620.0, 20.0, 31.0
    derivative = lambda temp: thermal_derivative(temp, np.zeros(1), np.zeros(1), np.array([inlet]), np.array([conductance]), np.zeros((1, 1)), np.array([capacitance]))
    duration = 1800.0
    reference = inlet + (initial - inlet) * np.exp(-conductance * duration / capacitance)
    def result(dt):
        state = np.array([initial])
        for _ in range(int(duration / dt)):
            state = rk4_step(state, dt, derivative)
        return state[0]
    assert abs(result(15) - reference) < abs(result(60) - reference)
    assert abs(result(15) - reference) < 1e-8


def test_symmetric_coupling_cancels_internal_energy():
    coupling = chain_coupling(4, np.zeros(4, dtype=int), 50.0)
    temperatures = np.array([20.0, 26.0, 31.0, 22.0])
    derivative = thermal_derivative(temperatures, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), coupling, np.ones(4))
    assert np.isclose(derivative.sum(), 0.0)
    assert np.allclose(coupling, coupling.T)


def test_actuator_lag_slew_and_capacity_are_real():
    actual = advance_actuator(np.array([20.0]), np.array([15.0]), tau_s=180, slew_per_s=1 / 60, dt_s=60)
    assert 15 < actual[0] < 20
    conductance, extracted = capacity_limited_conductance(np.array([60.0, 60.0]), np.array([20.0, 20.0]), 1000, np.array([1.0]), np.array([0, 0]), 10000, 1.0)
    assert extracted[0] <= 10000.0001
    assert np.all(conductance < 1000)


def test_plant_outputs_finite_without_temperature_clipping():
    config = load_config(profile="smoke")
    plant = DataCenterPlant(config, seed=10)
    for _ in range(10):
        snapshot = plant.advance(np.full(12, .65), 30.0, 50.0)
    assert np.all(np.isfinite(plant.hidden.temperatures_c))
    assert np.all(plant.hidden.temperatures_c < 100)
    assert snapshot.time_s == 600


def test_energy_conversion_is_kwh():
    assert integrate_energy_kwh(np.array([0.0, 3600.0]), np.array([1000.0, 1000.0])) == 1.0
