"""Lexical features computed from a hostname string only (no network, no page content).

Two scopes support the subdomain ablation described in the data audit:
- "host":       features of the full hostname plus explicit subdomain features;
- "registered": the same lexical features on the registered domain only, so no
                feature can see the subdomain at all.

Keyword lists are fixed a priori from common phishing vocabulary, not tuned on this data.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd


SENSITIVE_WORDS = (
    "login", "signin", "logon", "verify", "secure", "account", "update", "confirm",
    "auth", "wallet", "support", "recover", "unlock", "billing", "payment", "bank",
    "webscr", "password",
)
BRANDS = (
    "paypal", "apple", "microsoft", "office", "outlook", "netflix", "amazon", "facebook",
    "instagram", "whatsapp", "google", "chase", "wellsfargo", "metamask", "coinbase",
    "binance", "dhl", "usps", "fedex", "ebay",
)
VOWELS = set("aeiou")

LEXICAL_FEATURES = [
    "host_len", "n_labels", "n_hyphens", "n_digits", "digit_ratio", "entropy",
    "max_label_len", "domain_label_len", "vowel_ratio", "max_consonant_run",
    "is_ip", "has_punycode", "n_sensitive_words", "brand_not_owner",
]
SUBDOMAIN_FEATURES = ["has_subdomain", "subdomain_labels", "subdomain_len"]


def shannon_entropy(text):
    """Bits per character: 0 for 'aaaa', 1 for 'ab', higher for random-looking strings."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def max_consonant_run(text):
    best = run = 0
    for ch in text:
        run = run + 1 if ch.isalpha() and ch not in VOWELS else 0
        best = max(best, run)
    return best


def lexical_features(name, registered_domain, suffix, is_ip):
    """Features of one string `name` (a full host or a registered domain)."""
    labels = name.split(".")
    letters = [ch for ch in name if ch.isalpha()]
    digits = sum(ch.isdigit() for ch in name)
    # Main label of the registered domain: "abcbank" in "abcbank.com.vn"; empty for IPs.
    owner = "" if is_ip else registered_domain[: len(registered_domain) - len(suffix)].rstrip(".")
    brands = [b for b in BRANDS if b in name]
    return {
        "host_len": len(name),
        "n_labels": len(labels),
        "n_hyphens": name.count("-"),
        "n_digits": digits,
        "digit_ratio": digits / len(name) if name else 0.0,
        "entropy": shannon_entropy(name),
        "max_label_len": max(map(len, labels)),
        "domain_label_len": len(owner),
        "vowel_ratio": sum(ch in VOWELS for ch in letters) / len(letters) if letters else 0.0,
        "max_consonant_run": max_consonant_run(name),
        "is_ip": int(is_ip),
        "has_punycode": int("xn--" in name),
        "n_sensitive_words": sum(word in name for word in SENSITIVE_WORDS),
        # A brand name in the host that is not the domain owner, e.g. paypal-secure.com.
        "brand_not_owner": int(any(b != owner for b in brands)),
    }


def extract_features(hosts, scope="host"):
    """Feature matrix (DataFrame) for rows of data/processed/hosts.csv."""
    if scope not in ("host", "registered"):
        raise ValueError("scope must be 'host' or 'registered'")
    hosts = hosts.fillna({"subdomain": "", "suffix": ""})
    rows = []
    for host, registered, subdomain, suffix, ip in zip(
            hosts["host"], hosts["registered_domain"], hosts["subdomain"], hosts["suffix"], hosts["is_ip"]):
        name = host if scope == "host" else registered
        row = lexical_features(name, registered, suffix, bool(ip))
        if scope == "host":
            row["has_subdomain"] = int(bool(subdomain))
            row["subdomain_labels"] = len(subdomain.split(".")) if subdomain else 0
            row["subdomain_len"] = len(subdomain)
        rows.append(row)
    columns = LEXICAL_FEATURES + (SUBDOMAIN_FEATURES if scope == "host" else [])
    return pd.DataFrame(rows, index=hosts.index, columns=columns).astype(np.float64)
