"""Load the exported RBF SVM and explain a single prediction (used by demo/server.py).

The explanation is SVM-native: f(x) = Σ α_i y_i K(x_i, x) + b is a sum over support
vectors, so the terms α_i y_i K(x_i, x) show which training hostnames pull the input
towards phishing (+) or legitimate (−).
"""

import json

import numpy as np
import pandas as pd

from src.data import ROOT, hostname, is_ip, split_domain, strip_www
from src.features import extract_features
from src.preprocessing import transform_counts, transform_indices
from src.smo import kernel_matrix


MODEL_NPZ = ROOT / "models" / "svm_rbf.npz"
MODEL_JSON = ROOT / "models" / "svm_rbf.json"
MAX_INPUT = 2_048


def parse_host(text):
    """Hostname exactly as the training data saw it: no scheme, userinfo, port, path or 'www.'."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter a URL or a hostname.")
    if len(text) > MAX_INPUT:
        raise ValueError("Input is too long.")
    try:
        host = strip_www(hostname(text if "://" in text else "http://" + text))
    except ValueError as error:          # e.g. malformed IPv6 brackets
        raise ValueError(f"Could not parse a hostname: {error}") from None
    if not host.isascii():
        # The dataset stores internationalised domains as punycode (xn--…); a browser does the same.
        # Without this, a homograph such as Cyrillic "аррӏе.com" would never trigger has_punycode.
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError("Could not convert the internationalised hostname to punycode.") from None
    if not host or ("." not in host and not is_ip(host)) or len(host) > 253:
        raise ValueError("Could not find a valid hostname (expected something like example.com).")
    return host


def zone(decision):
    """Three-level policy from the SVM margin: outside the margin = confident, inside = uncertain."""
    if decision >= 1:
        return "phishing"
    if decision <= -1:
        return "legitimate"
    return "uncertain"


class DemoModel:
    def __init__(self, npz_path=MODEL_NPZ, json_path=MODEL_JSON):
        if not npz_path.exists() or not json_path.exists():
            raise FileNotFoundError("Demo model not found: run `python -m scripts.train_demo_model` first.")
        with np.load(npz_path, allow_pickle=False) as data:
            self.mean, self.scale = data["mean"], data["scale"]
            self.support_vectors, self.dual_coef = data["support_vectors"], data["dual_coef"]
        self.meta = json.loads(json_path.read_text(encoding="utf-8"))
        self.gamma, self.intercept = self.meta["gamma"], self.meta["intercept"]

    def summary(self):
        keys = ("solver", "kernel", "C", "gamma", "n_train", "n_test", "smo_iterations", "test_metrics", "features")
        return {**{k: self.meta[k] for k in keys}, "n_support": len(self.dual_coef),
                "preprocessing": self.meta.get("preprocessing", "standard"),
                "bias_dominated_test": self.meta.get("bias_dominated_test")}

    def predict(self, text, top_k=5):
        host = parse_host(text)
        registered, subdomain, suffix = split_domain(host)
        row = pd.DataFrame([{"host": host, "registered_domain": registered, "subdomain": subdomain,
                             "suffix": suffix, "is_ip": int(is_ip(host))}])
        x = extract_features(row).to_numpy()[0]
        log_idx, binarize_idx = transform_indices(self.meta.get("preprocessing", "standard"), self.meta["features"])
        z = (transform_counts(x[None, :], log_idx, binarize_idx)[0] - self.mean) / self.scale
        similarity = kernel_matrix(z[None, :], self.support_vectors, "rbf", self.gamma)[0]
        terms = self.dual_coef * similarity                     # α_i y_i K(x_i, x)
        decision = float(terms.sum() + self.intercept)
        top = np.argsort(-np.abs(terms))[:top_k]
        # RBF blind spot: far from every support vector, K ≈ 0 and f(x) collapses to the bias b.
        evidence = float(np.abs(terms).sum())
        out_of_distribution = evidence < abs(self.intercept)
        return {
            "input": text.strip(), "host": host, "registered_domain": registered,
            "subdomain": subdomain, "suffix": suffix,
            "decision": decision, "label": "phishing" if decision > 0 else "legitimate", "zone": zone(decision),
            "pull_phishing": float(terms[terms > 0].sum()), "pull_legitimate": float(terms[terms < 0].sum()),
            "intercept": self.intercept, "max_similarity": float(similarity.max()),
            "out_of_distribution": out_of_distribution,
            "extreme_features": [{"name": n, "z": float(s)} for n, s in zip(self.meta["features"], z) if abs(s) >= 4],
            "features": [{"name": n, "value": float(v), "z": float(s)}
                         for n, v, s in zip(self.meta["features"], x, z)],
            "nearest_support_vectors": [{
                "host": self.meta["support_hosts"][i],
                "label": "phishing" if self.meta["support_labels"][i] == 1 else "legitimate",
                "similarity": float(similarity[i]), "contribution": float(terms[i]),
            } for i in top],
        }
