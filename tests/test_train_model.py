"""Regression tests for train_model.py: the retry helper, train(), and save()."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import train_model

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace time.sleep in train_model with a recorder so tests run instantly."""
    sleeps = []
    monkeypatch.setattr(train_model.time, "sleep", sleeps.append)
    return sleeps


# ---------------------------------------------------------------------------
# retry()
# ---------------------------------------------------------------------------


def test_retry_returns_result_on_first_success(no_sleep):
    assert train_model.retry(lambda: "ok") == "ok"
    assert no_sleep == []


def test_retry_recovers_after_transient_failures(no_sleep):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("disk busy")
        return "ok"

    assert train_model.retry(flaky) == "ok"
    assert calls["n"] == 3
    assert no_sleep == [1.0, 2.0]


def test_retry_uses_exponential_backoff_with_cap(no_sleep):
    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        train_model.retry(always_fail, max_attempts=6, base_delay=1.0, max_delay=5.0)

    # 5 sleeps between 6 attempts: 1, 2, 4, then capped at 5.
    assert no_sleep == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_retry_reraises_last_exception_after_max_attempts(no_sleep):
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise ValueError(f"attempt {calls['n']}")

    with pytest.raises(ValueError, match="attempt 3"):
        train_model.retry(always_fail, max_attempts=3)

    assert calls["n"] == 3
    assert len(no_sleep) == 2


def test_retry_single_attempt_does_not_sleep(no_sleep):
    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        train_model.retry(always_fail, max_attempts=1)

    assert no_sleep == []


def test_retry_logs_attempts(no_sleep, capsys):
    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        train_model.retry(always_fail, max_attempts=2)

    out = capsys.readouterr().out
    assert "Attempt 1/2 failed: boom. Retrying in 1.0s..." in out
    assert "Attempt 2/2 failed: boom. Giving up." in out


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_returns_fitted_random_forest():
    model = train_model.train()
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_"), "model should be fitted"
    assert model.n_features_in_ == 4
    assert list(model.classes_) == [0, 1, 2]


def test_train_model_predicts_known_iris_samples():
    model = train_model.train()
    # Canonical samples: one clear example of each iris species.
    samples = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.0, 2.9, 4.5, 1.5],  # versicolor
            [6.9, 3.1, 5.4, 2.1],  # virginica
        ]
    )
    assert list(model.predict(samples)) == [0, 1, 2]


def test_train_accuracy_on_training_data_is_high():
    from sklearn.datasets import load_iris

    data = load_iris()
    model = train_model.train()
    assert model.score(data.data, data.target) >= 0.95


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_writes_loadable_pickle(tmp_path):
    model = train_model.train()
    path = tmp_path / "model.pkl"

    train_model.save(model, path)

    assert path.exists()
    assert path.stat().st_size > 0
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, RandomForestClassifier)
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    assert loaded.predict(sample) == model.predict(sample)


def test_save_overwrites_existing_file(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"stale contents")

    train_model.save(train_model.train(), path)

    with open(path, "rb") as f:
        assert isinstance(pickle.load(f), RandomForestClassifier)


def test_save_default_path_is_model_pkl():
    assert train_model.MODEL_PATH == "model.pkl"


def test_save_raises_when_directory_missing(tmp_path):
    missing = tmp_path / "does-not-exist" / "model.pkl"
    with pytest.raises(FileNotFoundError):
        train_model.save(train_model.train(), missing)


def test_save_is_retried_on_transient_io_error(tmp_path, no_sleep, monkeypatch):
    path = tmp_path / "model.pkl"
    model = train_model.train()
    real_open = open
    calls = {"n": 0}

    def flaky_open(file, *args, **kwargs):
        if str(file) == str(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk busy")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)

    train_model.retry(lambda: train_model.save(model, path))

    assert calls["n"] == 2
    assert no_sleep == [1.0]
    with real_open(path, "rb") as f:
        assert isinstance(pickle.load(f), RandomForestClassifier)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def test_script_end_to_end_writes_model_pkl(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "train_model.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Model saved!" in result.stdout
    with open(tmp_path / "model.pkl", "rb") as f:
        assert isinstance(pickle.load(f), RandomForestClassifier)
