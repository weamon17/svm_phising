import math

import pandas as pd
import pytest

from src.data import split_domain
from src.features import LEXICAL_FEATURES, SUBDOMAIN_FEATURES, extract_features, shannon_entropy


def hosts_frame(*hosts):
    rows = []
    for host in hosts:
        registered, subdomain, suffix = split_domain(host)
        rows.append({"host": host, "registered_domain": registered, "subdomain": subdomain,
                     "suffix": suffix, "is_ip": int(registered == host and suffix == "")})
    return pd.DataFrame(rows)


def test_entropy_known_values():
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("ab") == 1
    assert math.isclose(shannon_entropy("abcd"), 2)


def test_hand_computed_phishing_host():
    # The worked example of the report: registered domain secure-login.xyz, subdomain abcbank-xacminh.
    f = extract_features(hosts_frame("abcbank-xacminh.secure-login.xyz")).iloc[0]
    assert f["host_len"] == 32 and f["n_labels"] == 3 and f["n_hyphens"] == 2
    assert f["n_digits"] == 0 and f["max_label_len"] == 15
    assert f["domain_label_len"] == len("secure-login")
    assert f["n_sensitive_words"] == 3          # bank, secure, login
    assert f["brand_not_owner"] == 0
    assert (f["has_subdomain"], f["subdomain_labels"], f["subdomain_len"]) == (1, 1, 15)
    assert f["max_consonant_run"] == 3          # "bcb" in abcbank, "xyz"
    assert math.isclose(f["vowel_ratio"], 9 / 28)  # 28 letters, vowels a,a,a,i,e,u,e,o,i


def test_brand_only_counts_when_not_the_owner():
    f = extract_features(hosts_frame("paypal.com", "paypal-secure.com",
                                     "paypal.com-cgi-bin.webscr.dstech.us.com"))
    assert f["brand_not_owner"].tolist() == [0, 1, 1]


def test_ip_and_punycode():
    f = extract_features(hosts_frame("191.252.178.240", "xn--80aaa.xn--p1ai"))
    assert f.loc[0, "is_ip"] == 1 and f.loc[0, "domain_label_len"] == 0 and f.loc[0, "n_digits"] == 12
    assert f.loc[1, "has_punycode"] == 1


def test_registered_scope_cannot_see_subdomain():
    frame = hosts_frame("abcbank-xacminh.secure-login.xyz", "secure-login.xyz")
    f = extract_features(frame, scope="registered")
    assert list(f.columns) == LEXICAL_FEATURES
    assert f.iloc[0].equals(f.iloc[1])
    assert list(extract_features(frame).columns) == LEXICAL_FEATURES + SUBDOMAIN_FEATURES
    with pytest.raises(ValueError):
        extract_features(frame, scope="url")
