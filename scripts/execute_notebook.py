"""Execute a notebook in place with a project-local kernel (no global kernel registration)."""

import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main(relative_path):
    runtime = ROOT / ".jupyter_runtime"
    kernel_dir = runtime / "kernels" / "svm-phishing"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "SVM phishing", "language": "python",
    }), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(runtime)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")

    path = ROOT / relative_path
    notebook = nbformat.read(path, as_version=4)
    # No per-cell limit: in FULL mode the tuning cells of notebook 02 take ~25 minutes each.
    client = NotebookClient(notebook, timeout=None, startup_timeout=60, kernel_name="svm-phishing",
                            resources={"metadata": {"path": str(path.parent)}}, allow_errors=False)
    print(f"Executing {path.name} ...", flush=True)
    client.execute()
    notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Done, no errors.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_data_audit.ipynb")
