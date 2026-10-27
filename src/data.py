"""Audit helpers and the URL -> hostname cleaning step for PhiUSIIL.

Design decision (see notebooks/01_data_audit.ipynb): every legitimate URL in the
raw CSV is a bare homepage "https://www.<domain>", so scheme, "www." and path
carry the collection process rather than phishing behaviour. Cleaning keeps only
the hostname, which both classes genuinely have, and drops all 54 precomputed
columns (page-content features and URL features computed on the biased string).
"""

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "hosts.csv"

# Raw convention from UCI: 1 = legitimate, 0 = phishing. We model phishing as positive.
RAW_LEGITIMATE = 1

# Offline Public Suffix List snapshot bundled with tldextract: no network at runtime.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def load_raw(path=RAW_CSV):
    return pd.read_csv(path)


def hostname(url):
    """Real host the browser connects to: drops scheme, userinfo ("a@b"), port, case."""
    return (urlsplit(url).hostname or "").rstrip(".").lower()


def strip_www(host):
    return host[4:] if host.startswith("www.") else host


def is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_bare_https_www(url):
    """The raw legitimate template: https://www.<host> with no path, query or fragment."""
    parts = urlsplit(url)
    return (url.startswith("https://www.") and parts.path in ("", "/")
            and not parts.query and not parts.fragment)


def univariate_auc(features, y):
    """Direction-free ROC-AUC of each numeric column alone (0.5 = useless, 1.0 = perfect)."""
    from sklearn.metrics import roc_auc_score
    scores = {}
    for column in features.columns:
        auc = roc_auc_score(y, features[column])
        scores[column] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def split_domain(host):
    """(registered_domain, subdomain, suffix); IPs and unknown suffixes use the host itself."""
    if is_ip(host):
        return host, "", ""
    parts = _EXTRACT(host)
    registered = parts.registered_domain or host
    return registered, parts.subdomain, parts.suffix


def clean_hosts(raw):
    """Return (one row per hostname, step-by-step log of rows/hosts removed)."""
    log = [{"step": "raw rows", "rows": len(raw)}]

    df = raw[["URL", "label"]].drop_duplicates("URL")
    log.append({"step": "drop duplicated URLs", "rows": len(df), "removed": log[-1]["rows"] - len(df)})

    df = df.assign(host=df["URL"].map(hostname).map(strip_www))
    df = df[df["host"] != ""]
    log.append({"step": "drop URLs without hostname", "rows": len(df), "removed": log[-1]["rows"] - len(df)})

    per_host = df.groupby("host").agg(n_urls=("URL", "size"), n_labels=("label", "nunique"),
                                      label=("label", "first"))
    log.append({"step": "collapse to unique hostnames", "rows": len(per_host),
                "removed": len(df) - len(per_host)})

    conflicting = per_host["n_labels"] > 1
    per_host = per_host[~conflicting]
    log.append({"step": "drop hostnames labelled both ways", "rows": len(per_host),
                "removed": int(conflicting.sum())})

    hosts = per_host.reset_index()
    split = hosts["host"].map(split_domain)
    hosts["registered_domain"] = split.str[0]
    hosts["subdomain"] = split.str[1]
    hosts["suffix"] = split.str[2]
    hosts["is_ip"] = hosts["host"].map(is_ip).astype(int)
    hosts["phishing"] = (hosts["label"] != RAW_LEGITIMATE).astype(int)
    hosts = hosts[["host", "registered_domain", "subdomain", "suffix", "is_ip", "n_urls", "phishing"]]
    return hosts.sort_values("host", ignore_index=True), log


def class_balance(labels):
    counts = labels.value_counts().sort_index()
    return pd.DataFrame({"count": counts, "share": np.round(counts / counts.sum(), 4)})
