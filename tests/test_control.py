"""Workload conservation, constraints, candidates, and safety-first controller tests."""

import numpy as np

from thermotwin.config import load_config
from thermotwin.control.actions import generate_candidates
from thermotwin.control.constraints import validate_action
from thermotwin.control.optimizer import optimize
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.workload import allocate
from thermotwin.types import Action, RolloutResult


def test_migration_conserves_served_work_and_persists_with_cooldown():
    config=load_config(profile="smoke"); plant=DataCenterPlant(config,seed=2); plant.offered_utilization=np.array([.9,.2]+[.5]*10); plant.utilization,_=allocate(plant.offered_utilization)
    before=plant.utilization.sum(); action=Action("move",(0,0,0),(0,0,0),(0,1,.1)); ok,reason=plant.apply_action(action)
    assert ok,reason
    plant.advance(plant.offered_utilization,27,50)
    assert np.isclose(plant.utilization.sum(),before)
    assert np.isclose(plant.placement_offset.sum(),0)
    assert plant.migration_cooldowns[0] > 0


def test_allocation_exposes_overload_without_dropping_hidden_work():
    served,unserved=allocate(np.array([1.4,1.2]),np.array([-.1,.1]))
    assert np.isclose(served.sum()+unserved,2.6)
    assert np.all(served<=1)


def test_candidate_set_includes_required_categories():
    config=load_config(profile="smoke"); plant=DataCenterPlant(config,seed=2); plant.utilization=np.linspace(.9,.2,12); snapshot=plant.snapshot(); ids=[a.action_id for a in generate_candidates(snapshot,config)]
    assert "noop" in ids and "maximum-safe-cooling" in ids
    assert any(value.startswith("supply-") for value in ids)
    assert any(value.startswith("air-") for value in ids)
    assert any(value.startswith("migrate-") for value in ids)


def test_invalid_nonfinite_action_is_rejected():
    config=load_config(profile="smoke"); snapshot=DataCenterPlant(config,2).snapshot(); valid,reason=validate_action(snapshot,Action("bad",(np.nan,0,0),(0,0,0)),config)
    assert not valid and reason=="nonfinite_command"


def test_optimizer_prefers_feasible_before_lower_soft_cost():
    config=load_config(profile="smoke"); snapshot=DataCenterPlant(config,2).snapshot(); safe=Action("safe",(0,0,0),(0,0,0)); unsafe=Action("unsafe",(-.5,0,0),(0,0,0))
    class Twin:
        def simulate_action(self,snapshot,action,*args,**kwargs):
            hot=41 if action.action_id=="unsafe" else 34; temps=np.full((2,12),hot); return RolloutResult(np.array([60,120]),temps,temps,np.ones(2),.01 if action.action_id=="unsafe" else 100,snapshot,hot<=40,hot,0,1 if hot>40 else 0)
    decision=optimize(snapshot,[unsafe,safe],Twin(),config,"test",False)
    assert decision.chosen_action.action_id=="safe"


def test_all_unsafe_uses_deterministic_least_violation_fallback():
    config=load_config(profile="smoke"); snapshot=DataCenterPlant(config,2).snapshot(); worse=Action("worse",(0,0,0),(0,0,0)); better=Action("better",(-.5,0,0),(0,0,0))
    class Twin:
        def simulate_action(self,snapshot,action,*args,**kwargs):
            hot=44 if action.action_id=="worse" else 42; temps=np.full((2,12),hot); return RolloutResult(np.array([60,120]),temps,temps,np.ones(2),1,snapshot,False,hot,10,2 if hot==44 else 1)
    decision=optimize(snapshot,[worse,better],Twin(),config,"test",False)
    assert decision.fallback and decision.chosen_action.action_id=="better"
