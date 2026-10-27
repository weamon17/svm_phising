"""Train/test split grouped by registered domain, then stratified subsampling for kernel SVMs."""

from sklearn.model_selection import GroupShuffleSplit, train_test_split


def group_split(hosts, test_size=0.2, seed=42):
    """Indices so that no registered domain (e.g. web.app) appears in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(hosts, hosts["phishing"], groups=hosts["registered_domain"]))
    return hosts.index[train_idx], hosts.index[test_idx]


def stratified_sample(index, labels, n, seed=42):
    """n rows of `index` keeping the class ratio of `labels` (a Series indexed like hosts)."""
    if n >= len(index):
        return index
    sample, _ = train_test_split(index, train_size=n, stratify=labels.loc[index], random_state=seed)
    return sample.sort_values()
