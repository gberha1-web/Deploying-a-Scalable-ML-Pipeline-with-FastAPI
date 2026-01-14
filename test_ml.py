import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics



@pytest.fixture(scope="module")
def census_df():
    """Load a small subset of the census data for tests."""

    repo_root = Path(__file__).resolve().parent
    data_path = repo_root / "data" / "census.csv"
    df = pd.read_csv(data_path)
    
    return df.sample(n=200, random_state=42).reset_index(drop=True)

    
@pytest.fixture(scope="module")
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
# TODO: implement the first test. Change the function name and input as needed

def test_process_data_outputs(census_df, cat_features):
    """process_data should return X, y with expected lengths and fitted encoder/lb."""
    X, y, encoder, lb = process_data(
        census_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # X can be numpy array or scipy sparse matrix depending on encoder settings
    assert isinstance(X, (np.ndarray, sparse.spmatrix))
    assert y is not None
    assert len(y) == len(census_df)

    # Encoder and label binarizer should be fitted
    assert hasattr(encoder, "categories_")
    assert hasattr(lb, "classes_")
    assert len(lb.classes_) == 2
    
   


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_logreg(census_df, cat_features):
    """train_model should return a LogisticRegression model and be usable for inference."""
    X, y, _, _ = process_data(
        census_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)

    preds = inference(model, X)
    assert len(preds) == len(y)

# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_perfect_predictions():
    """compute_model_metrics should return 1.0s for perfect prediction."""
    y = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0