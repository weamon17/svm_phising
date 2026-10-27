import pandas as pd

from src.data import clean_hosts, hostname, is_bare_https_www, split_domain, strip_www


def test_hostname_ignores_userinfo_port_and_case():
    # Classic trick: everything before "@" is userinfo, the real host comes after it.
    assert hostname("http://www.paypal.com:911@xn--80aaa.xn--p1ai/fix/") == "xn--80aaa.xn--p1ai"
    assert hostname("https://Shop.Example.COM:443/a?b=1") == "shop.example.com"
    assert hostname("http://47.103.128.80:8085/din.php") == "47.103.128.80"


def test_strip_www_only_removes_leading_label():
    assert strip_www("www.bbc.co.uk") == "bbc.co.uk"
    assert strip_www("www-bank.com") == "www-bank.com"
    assert strip_www("a.www.example.com") == "a.www.example.com"


def test_bare_https_www_template():
    assert is_bare_https_www("https://www.uni-mainz.de")
    assert is_bare_https_www("https://www.uni-mainz.de/")
    assert not is_bare_https_www("http://www.uni-mainz.de")
    assert not is_bare_https_www("https://uni-mainz.de")
    assert not is_bare_https_www("https://www.uni-mainz.de/login")
    assert not is_bare_https_www("https://www.uni-mainz.de?id=1")


def test_split_domain_handles_multi_part_suffix_and_ip():
    assert split_domain("a.b.firebaseapp.com") == ("firebaseapp.com", "a.b", "com")
    assert split_domain("bbc.co.uk") == ("bbc.co.uk", "", "co.uk")
    assert split_domain("191.252.178.240") == ("191.252.178.240", "", "")


def test_clean_hosts_dedups_collapses_and_drops_conflicts():
    raw = pd.DataFrame({
        "URL": [
            "https://www.good.com",            # legit
            "https://www.good.com",            # exact duplicate URL
            "https://evil.web.app/login",      # phishing
            "http://evil.web.app/verify",      # same host, same label -> collapsed
            "https://www.mixed.org",           # legit ...
            "http://mixed.org/phish",          # ... and phishing on the same host -> dropped
        ],
        "label": [1, 1, 0, 0, 1, 0],
    })
    hosts, log = clean_hosts(raw)

    assert hosts["host"].tolist() == ["evil.web.app", "good.com"]
    evil = hosts.set_index("host").loc["evil.web.app"]
    assert evil["phishing"] == 1 and evil["n_urls"] == 2
    assert evil["registered_domain"] == "web.app" and evil["subdomain"] == "evil"
    assert hosts.set_index("host").loc["good.com", "phishing"] == 0
    assert [entry["rows"] for entry in log] == [6, 5, 5, 3, 2]
