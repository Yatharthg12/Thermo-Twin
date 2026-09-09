"""Known-trajectory event, duration, energy, PUE, and uncertainty statistics."""

import numpy as np

from thermotwin.experiments.metrics import duration_rack_minutes, hotspot_events, pue_approximation, relative_savings
from thermotwin.experiments.statistics import paired_bootstrap_interval


def test_contiguous_events_distinguish_boundary_active():
    temps=np.array([[36,30],[37,36],[34,37],[36,34]],float); events,boundary=hotspot_events(np.arange(4)*60,temps,35)
    assert events==3 and boundary==1


def test_irregular_duration_uses_elapsed_intervals():
    temps=np.array([[30],[36],[36]],float); assert duration_rack_minutes(np.array([0,30,120]),temps,35)==2.0


def test_pue_and_savings_guard_zero_denominators():
    assert pue_approximation(10,4)["value"]==1.4
    assert pue_approximation(0,4)["value"] is None
    assert relative_savings(1,0,"fixed")["value"] is None


def test_bootstrap_requires_independent_replications():
    assert paired_bootstrap_interval(np.array([1,2]),np.array([1,1]))["ci95"] is None
    assert paired_bootstrap_interval(np.array([1,2,3]),np.array([.9,1.8,2.7]))["ci95"] is not None

