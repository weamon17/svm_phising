import pickle

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.features import LEXICAL_FEATURES, SUBDOMAIN_FEATURES
from src.preprocessing import (bias_dominated_rate, make_preprocessor, one_standard_error_choice,
                               transform_counts, transform_indices)


HOST = LEXICAL_FEATURES + SUBDOMAIN_FEATURES


def test_transform_counts_touches_only_selected_columns_and_keeps_order():
    X = np.array([[0.0, 3.0, 0.5], [9.0, 0.0, 0.25]])
    out = transform_counts(X, log_idx=[0], binarize_idx=[1])
    np.testing.assert_allclose(out[:, 0], np.log1p([0.0, 9.0]))
    np.testing.assert_array_equal(out[:, 1], [1.0, 0.0])
    np.testing.assert_array_equal(out[:, 2], X[:, 2])
    assert X[0, 1] == 3.0                                        # input not modified


def test_variants_select_the_right_columns():
    log, binarize = transform_indices("log1p+binarize", HOST)
    assert binarize == [HOST.index("n_sensitive_words")]
    assert HOST.index("n_sensitive_words") not in log and HOST.index("host_len") in log
    assert HOST.index("entropy") not in log and HOST.index("is_ip") not in log   # continuous / binary untouched
    assert transform_indices("standard", HOST) == ([], [])
    log, binarize = transform_indices("log1p", LEXICAL_FEATURES)                  # registered scope: no subdomain cols
    assert binarize == [] and len(log) == 8
    with pytest.raises(ValueError):
        transform_indices("quantile", HOST)


def test_pipeline_is_picklable_for_parallel_grid_search():
    pipe = make_pipeline(*make_preprocessor("log1p+binarize", HOST), SVC())
    assert pickle.loads(pickle.dumps(pipe)) is not None


def test_one_standard_error_rule_keeps_simplest_unless_clearly_better():
    import pandas as pd
    def frame(cv, std):
        return pd.DataFrame({"model": "SVC rbf", "preprocessing": ["standard", "log1p", "log1p+binarize"],
                             "cv_f1": cv, "cv_f1_std": std})
    # FULL run: 0.701 vs 0.703 with fold std ≈ 0.10 → noise, keep "standard".
    assert one_standard_error_choice(frame([0.701, 0.703, 0.703], [0.10] * 3), 5)[0] == "standard"
    # A gain larger than one standard error (0.01 / √5 ≈ 0.0045) is kept.
    assert one_standard_error_choice(frame([0.70, 0.75, 0.74], [0.01] * 3), 5)[0] == "log1p"


def test_bias_dominated_rate_flags_far_points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipe = make_pipeline(*make_preprocessor("standard", ["a", "b"]), SVC(kernel="rbf", gamma=1.0)).fit(X, y)
    assert bias_dominated_rate(pipe, X) < 0.1
    assert bias_dominated_rate(pipe, X + 1e3) == 1.0
