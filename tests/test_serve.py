import importlib
import sys

import joblib
import numpy as np
import pytest
from fastapi import HTTPException
from google.cloud import storage


class _DummyModel:
    def predict(self, features):
        assert len(features[0]) == 12
        return np.array([2])


class _DummyBlob:
    def download_to_filename(self, path):
        joblib.dump(_DummyModel(), path)


class _DummyBucket:
    def blob(self, key):
        assert key == "models/latest/model.pkl"
        return _DummyBlob()


class _DummyClient:
    def bucket(self, name):
        assert name == "test-bucket"
        return _DummyBucket()


@pytest.fixture
def serve_module(monkeypatch, tmp_path):
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(storage, "Client", _DummyClient)
    monkeypatch.setattr(
        "os.path.expanduser",
        lambda _path: str(tmp_path / "model.pkl"),
    )

    sys.modules.pop("src.serve", None)
    return importlib.import_module("src.serve")


def test_health(serve_module):
    assert serve_module.health() == {"status": "ok"}


def test_predict(serve_module):
    request = serve_module.PredictRequest(features=[0.0] * 12)
    assert serve_module.predict(request) == {"prediction": 2, "label": "cao"}


def test_predict_rejects_wrong_feature_count(serve_module):
    request = serve_module.PredictRequest(features=[0.0] * 11)
    with pytest.raises(HTTPException) as exc_info:
        serve_module.predict(request)
    assert exc_info.value.status_code == 400
