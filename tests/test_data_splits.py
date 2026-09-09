"""Temporal partition, purge, and causal feature invariants."""

import numpy as np
import pandas as pd
import pytest

from thermotwin.config import load_config
from thermotwin.data.features import build_supervised
from thermotwin.data.splitting import assert_partition_local, chronological_ranges


def _frame(n_steps=600, racks=2):
    rows=[]
    for step in range(n_steps):
        for rack in range(racks):
            temp=25 + .005*step + .1*rack
            rows.append({"run_id":"r1","step":step,"rack_id":rack,"estimated_temp_c":temp,"true_temp_c":temp+.05,"utilization":.5,"supply_actual_c":20.,"airflow_actual":.9,"ambient_c":27.,"zone_id":rack})
    return pd.DataFrame(rows)


def _config():
    config=load_config(profile="smoke")
    config["topology"]={"racks":2,"zones":2,"rack_zone":[0,1]}
    return config


def test_ranges_have_full_purge_gaps():
    ranges=chronological_ranges(600,12,60)
    for left,right in zip(ranges,ranges[1:]):
        assert right.start-left.end == 72


def test_partition_local_guard_rejects_crossing_target():
    split=chronological_ranges(600,12,60)[0]
    with pytest.raises(ValueError):
        assert_partition_local(split.start, split.end-2, split.end+3, split)


def test_future_test_perturbation_cannot_change_train_features_or_targets():
    frame=_frame()
    parts,_=build_supervised(frame,_config())
    changed=frame.copy()
    test_start=chronological_ranges(600,12,60)[-1].start
    changed.loc[changed.step>=test_start,"true_temp_c"] += 100
    changed_parts,_=build_supervised(changed,_config())
    assert np.array_equal(parts["train"].X,changed_parts["train"].X)
    assert np.array_equal(parts["train"].y,changed_parts["train"].y)


def test_windows_have_expected_shapes_and_stay_in_run():
    parts,manifest=build_supervised(_frame(),_config())
    assert parts["train"].X.shape[1] == 15
    assert parts["train"].X_sequence.shape[1:] == (12,7)
    assert parts["train"].y.shape[1] == 5
    assert parts["train"].hotspot_events.shape[1] == 5
    assert manifest[0]["purge_steps"] == 72
