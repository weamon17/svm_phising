"""Download the unmodified PhiUSIIL CSV from UCI and record provenance."""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
LANDING_URL = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


def main():
    destination = ROOT / "data" / "raw" / CSV_NAME
    metadata_path = destination.with_suffix(".source.json")
    if destination.exists():
        if not metadata_path.exists():
            raise FileExistsError("CSV exists without provenance; inspect it before replacing.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != metadata["csv_sha256"]:
            raise ValueError("Local CSV differs from its recorded SHA-256.")
        print(f"Verified existing dataset: {destination}")
        return

    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "SVMPhishingCourseProject/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        matches = [n for n in archive.namelist() if Path(n).name == CSV_NAME]
        if len(matches) != 1:
            raise ValueError(f"Expected one CSV named {CSV_NAME}, archive has {archive.namelist()}")
        # Read only the selected member; do not extract arbitrary archive paths.
        csv_bytes = archive.read(matches[0])

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    metadata = {
        "source_url": SOURCE_URL,
        "landing_url": LANDING_URL,
        "paper_doi": "10.1016/j.cose.2023.103545",
        "citation": (
            "Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered "
            "phishing URL detection framework based on similarity index and incremental learning. "
            "Computers & Security, 136, 103545."
        ),
        "license": "CC BY 4.0",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "csv_bytes": len(csv_bytes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Downloaded {len(csv_bytes):,} bytes to {destination}")


if __name__ == "__main__":
    main()
