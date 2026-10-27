import pickle
import time

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"


def retry(func, max_attempts=5, base_delay=1.0, max_delay=30.0, backoff_factor=2.0):
    """Call func() and retry on failure with exponential backoff.

    Delay before attempt n (n >= 2) is min(base_delay * backoff_factor ** (n - 2), max_delay).
    With the defaults that is 1s, 2s, 4s, 8s between the five attempts.
    Re-raises the last exception once max_attempts is exhausted.
    """
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt == max_attempts:
                print(f"Attempt {attempt}/{max_attempts} failed: {exc}. Giving up.")
                raise
            print(f"Attempt {attempt}/{max_attempts} failed: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)


def train():
    data = load_iris()
    model = RandomForestClassifier()
    model.fit(data.data, data.target)
    return model


def save(model, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    model = retry(train)
    retry(lambda: save(model))
    print("Model saved!")
