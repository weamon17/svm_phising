"""Build data/processed/hosts.csv from the verified raw CSV and log every removal."""

import hashlib
import json
from datetime import datetime, timezone

from src.data import PROCESSED_CSV, RAW_CSV, clean_hosts, load_raw


def main():
    metadata = json.loads(RAW_CSV.with_suffix(".source.json").read_text(encoding="utf-8"))
    if hashlib.sha256(RAW_CSV.read_bytes()).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("Raw CSV differs from its recorded SHA-256; re-run scripts/download_data.py.")

    hosts, log = clean_hosts(load_raw())
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    hosts.to_csv(PROCESSED_CSV, index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_csv_sha256": metadata["csv_sha256"],
        "steps": log,
        "phishing_hosts": int(hosts["phishing"].sum()),
        "legitimate_hosts": int((hosts["phishing"] == 0).sum()),
        "registered_domains": int(hosts["registered_domain"].nunique()),
        "processed_sha256": hashlib.sha256(PROCESSED_CSV.read_bytes()).hexdigest(),
    }
    PROCESSED_CSV.with_name("cleaning_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for step in log:
        print(f"{step['step']:<36} rows={step['rows']:>7,}  removed={step.get('removed', 0):>6,}")
    print(f"phishing={summary['phishing_hosts']:,}  legitimate={summary['legitimate_hosts']:,}")


if __name__ == "__main__":
    main()
