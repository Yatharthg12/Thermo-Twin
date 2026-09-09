"""Required family fitting, recurrent behavior, calibration, and trusted persistence."""

import numpy as np
import pytest
import json

from thermotwin.config import load_config
from thermotwin.forecasting.classical import make_classical_models
from thermotwin.forecasting.hybrid import ActionTransitionModel, ResidualCorrector, fit_operational_hybrid
from thermotwin.forecasting.recurrent import train_gru
from thermotwin.forecasting.registry import load_bundle, save_bundle
from thermotwin.forecasting.uncertainty import HotspotCalibrator, IntervalCalibrator


def _arrays(seed=1):
    rng=np.random.default_rng(seed); X=rng.normal(size=(80,15)).astype("float32"); y=(X[:,:5]*.2+25).astype("float32"); seq=rng.normal(size=(80,12,7)).astype("float32"); return X,y,seq


def test_every_classical_family_really_fits_and_predicts_shape():
    config=load_config(profile="smoke"); X,y,_=_arrays()
    for model in make_classical_models(config,4).values():
        model.fit(X[:60],y[:60]); prediction=model.predict(X[60:]); assert prediction.shape==(20,5); assert np.isfinite(prediction).all()


def test_gru_is_trained_recurrent_model_with_best_checkpoint():
    config=load_config(profile="smoke"); _,y,seq=_arrays()
    trained=train_gru(seq[:60],y[:60],seq[60:],y[60:],config,5)
    assert trained.model.gru.__class__.__name__ == "GRU"
    assert trained.trained_epochs >= 1
    assert trained.predict(seq[60:]).shape == (20,5)


def test_save_load_prediction_equivalence(tmp_path):
    config=load_config(profile="smoke"); X,y,seq=_arrays(); classical=make_classical_models(config,3)
    for model in classical.values(): model.fit(X[:60],y[:60])
    recurrent=train_gru(seq[:60],y[:60],seq[60:],y[60:],config,3)
    nominal=np.repeat(X[:,[0]],5,axis=1); hybrid=fit_operational_hybrid(X[:60],y[:60],nominal[:60]); interval=IntervalCalibrator.fit(y[60:],nominal[60:]); events=(y[60:]>25).astype(int); hotspot=HotspotCalibrator(35).fit(nominal[60:],events,np.ones_like(y[60:]));
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    transition=ActionTransitionModel(Pipeline([("model",Ridge())]).fit(np.random.default_rng(2).normal(size=(80,9)),np.random.default_rng(3).normal(size=80)))
    residual=ResidualCorrector(Pipeline([("model",Ridge())]).fit(np.random.default_rng(4).normal(size=(80,8)),np.random.default_rng(5).normal(size=80)))
    bundle={"classical":classical,"recurrent":recurrent,"hybrid":hybrid,"interval":interval,"hotspot":hotspot,"action_residual":residual,"learned_transition":transition}
    save_bundle(tmp_path,bundle,{"horizons_min":[5,10,15,30,60]}); loaded,_=load_bundle(tmp_path)
    assert np.allclose(bundle["classical"]["ridge"].predict(X[60:]),loaded["classical"]["ridge"].predict(X[60:]))
    assert np.allclose(recurrent.predict(seq[60:]),loaded["recurrent"].predict(seq[60:]))


def test_package_stale_artifact_fails_before_unpickling(tmp_path):
    manifest={"schema":"1.0","feature_names":__import__("thermotwin.data.features",fromlist=["FEATURE_NAMES"]).FEATURE_NAMES,"packages":{"scikit_learn":"0.0-impossible"}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError,match="package versions"):
        load_bundle(tmp_path)


def test_missing_model_artifact_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError,match="prepare"):
        load_bundle(tmp_path)
