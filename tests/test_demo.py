import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from src.demo import DemoModel, parse_host, zone
from src.features import extract_features
from src.data import split_domain
from src.preprocessing import transform_counts, transform_indices
from src.smo import SMOSVC


def test_parse_host_accepts_urls_and_bare_hosts():
    assert parse_host("https://www.Example.com/login?x=1") == "example.com"
    assert parse_host("  shop.example.co.uk  ") == "shop.example.co.uk"
    assert parse_host("http://paypal.com@evil.example.ru/") == "evil.example.ru"
    assert parse_host("191.252.178.240:8087/x") == "191.252.178.240"


def test_parse_host_converts_unicode_homographs_to_punycode():
    # Cyrillic look-alike of apple.com must reach the model in the same xn-- form as the training data.
    assert parse_host("https://аррӏе.com/login") == "xn--80ak6aa92e.com"
    long_ascii_label = "a" * 70 + ".example.com"                       # IDNA would reject, ASCII must pass
    assert parse_host(long_ascii_label) == long_ascii_label


@pytest.mark.parametrize("bad", ["", "   ", "localhost", "http://", "x" * 3000])
def test_parse_host_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        parse_host(bad)


def test_zone_follows_the_margin():
    assert [zone(v) for v in (2.0, 1.0, 0.3, -0.3, -1.0, -2.0)] == [
        "phishing", "phishing", "uncertain", "uncertain", "legitimate", "legitimate"]


@pytest.fixture
def tiny_model(tmp_path):
    """A small exported model in the same format as scripts/train_demo_model.py."""
    hosts = ["a.example.com", "login-secure.web.app", "shop.example.org", "x1-verify.top",
             "news.site.net", "acc0unt-update.xyz", "blog.example.io", "pay-confirm.example.app"]
    frame = __import__("pandas").DataFrame([
        dict(zip(("registered_domain", "subdomain", "suffix"), split_domain(h)), host=h, is_ip=0) for h in hosts])
    X = extract_features(frame).to_numpy()
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scaler = StandardScaler().fit(X)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    model = SMOSVC(C=1.0, kernel="rbf", gamma=0.3).fit((X - scaler.mean_) / scale, y)
    np.savez(tmp_path / "m.npz", mean=scaler.mean_, scale=scale,
             support_vectors=model.support_vectors_, dual_coef=model.dual_coef_)
    meta = {"solver": "SMOSVC", "kernel": "rbf", "C": 1.0, "gamma": model.gamma_, "intercept": model.intercept_,
            "n_train": 8, "n_test": 0, "smo_iterations": model.n_iter_, "test_metrics": {},
            "features": list(extract_features(frame).columns),
            "support_hosts": [hosts[i] for i in model.support_],
            "support_labels": [int(y[i]) for i in model.support_]}
    (tmp_path / "m.json").write_text(json.dumps(meta), encoding="utf-8")
    return DemoModel(tmp_path / "m.npz", tmp_path / "m.json"), model, scaler, scale


def test_prediction_matches_smo_decision_function(tiny_model):
    demo, model, scaler, scale = tiny_model
    result = demo.predict("https://www.pay-verify.example.app/login")
    frame = __import__("pandas").DataFrame([dict(zip(("registered_domain", "subdomain", "suffix"),
                                                     split_domain(result["host"])), host=result["host"], is_ip=0)])
    expected = model.decision_function((extract_features(frame).to_numpy() - scaler.mean_) / scale)[0]
    assert result["host"] == "pay-verify.example.app"
    assert result["decision"] == pytest.approx(expected)
    # f(x) = Σ α_i y_i K(x_i, x) + b, split into the two pulls plus the bias.
    assert result["pull_phishing"] + result["pull_legitimate"] + result["intercept"] == pytest.approx(expected)
    assert result["label"] == ("phishing" if expected > 0 else "legitimate")
    assert len(result["features"]) == 17 and 1 <= len(result["nearest_support_vectors"]) <= 5
    assert result["out_of_distribution"] is False


def test_demo_applies_the_exported_preprocessing(tmp_path):
    """A model exported with --prep log1p+binarize must transform inputs the same way at prediction time."""
    hosts = ["a.example.com", "login-secure.web.app", "shop.example.org", "x1-verify.top",
             "news.site.net", "acc0unt-update.xyz", "blog.example.io", "pay-confirm.example.app"]
    frame = __import__("pandas").DataFrame([
        dict(zip(("registered_domain", "subdomain", "suffix"), split_domain(h)), host=h, is_ip=0) for h in hosts])
    names = list(extract_features(frame).columns)
    log_idx, binarize_idx = transform_indices("log1p+binarize", names)
    X = transform_counts(extract_features(frame).to_numpy(), log_idx, binarize_idx)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    mean, scale = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    model = SMOSVC(C=1.0, kernel="rbf", gamma=0.3).fit((X - mean) / scale, y)
    np.savez(tmp_path / "m.npz", mean=mean, scale=scale, support_vectors=model.support_vectors_,
             dual_coef=model.dual_coef_)
    (tmp_path / "m.json").write_text(json.dumps({
        "preprocessing": "log1p+binarize", "gamma": model.gamma_, "intercept": model.intercept_, "features": names,
        "support_hosts": [hosts[i] for i in model.support_], "support_labels": [int(y[i]) for i in model.support_],
    }), encoding="utf-8")
    demo = DemoModel(tmp_path / "m.npz", tmp_path / "m.json")

    result = demo.predict("verify-login-secure-update.example.app")
    row = __import__("pandas").DataFrame([dict(zip(("registered_domain", "subdomain", "suffix"),
                                                   split_domain(result["host"])), host=result["host"], is_ip=0)])
    x = transform_counts(extract_features(row).to_numpy(), log_idx, binarize_idx)
    assert result["decision"] == pytest.approx(model.decision_function((x - mean) / scale)[0])
    sensitive = next(f for f in result["features"] if f["name"] == "n_sensitive_words")
    assert sensitive["value"] == 4                                   # raw count is still shown to the user


def test_far_input_is_flagged_and_falls_back_to_bias(tiny_model):
    demo = tiny_model[0]
    result = demo.predict("-".join(["login9verify9secure"] * 10) + ".example.com")
    assert result["out_of_distribution"] is True
    assert result["max_similarity"] < 1e-6
    assert result["decision"] == pytest.approx(result["intercept"], abs=1e-6)
    assert result["extreme_features"]
