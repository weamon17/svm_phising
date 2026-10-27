import pandas as pd

from src.splits import group_split, stratified_sample


def toy_hosts():
    rows = []
    for g in range(40):
        for k in range(5):
            rows.append({"host": f"h{k}.d{g}.com", "registered_domain": f"d{g}.com", "phishing": int(g % 3 == 0)})
    return pd.DataFrame(rows)


def test_group_split_never_shares_a_registered_domain():
    hosts = toy_hosts()
    train, test = group_split(hosts, test_size=0.25, seed=0)
    assert len(train) + len(test) == len(hosts)
    assert not set(hosts.loc[train, "registered_domain"]) & set(hosts.loc[test, "registered_domain"])


def test_stratified_sample_keeps_class_ratio_and_is_reproducible():
    hosts = toy_hosts()
    sample = stratified_sample(hosts.index, hosts["phishing"], 60, seed=1)
    assert len(sample) == 60
    assert abs(hosts.loc[sample, "phishing"].mean() - hosts["phishing"].mean()) < 0.02
    assert sample.equals(stratified_sample(hosts.index, hosts["phishing"], 60, seed=1))
    assert stratified_sample(hosts.index, hosts["phishing"], 10_000).equals(hosts.index)
