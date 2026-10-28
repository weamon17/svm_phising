"""Build the graded submission folder and zip in dist/.

    python -m scripts.make_submission --ids 52200000 52200001 52200002 \
        --report BaoCao.docx BaoCao.pdf --slides Slide.pptx

Layout of dist/<IDs>/ (zipped as dist/<IDs>.zip, as the assignment requires):
    HUONG_DAN.md      how to grade and run (generated from scripts/submission_guide.md)
    BaoCao/           report .docx + .pdf
    Slide/            slides
    Notebooks_HTML/   both notebooks with their outputs, viewable in any browser
    SourceCode/       every file Git would commit (tracked + new, minus .gitignore), incl. the .bat launchers

Without --draft the build refuses to run while the report (.docx and .pdf) or slides are missing.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat

from src.data import ROOT


DIST = ROOT / "dist"
GUIDE_TEMPLATE = ROOT / "scripts" / "submission_guide.md"


def source_files():
    """Files Git would put in a commit right now: tracked and new, minus .gitignore."""
    listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(f for f in listing.split("\0") if f and (ROOT / f).is_file())


def uncommitted_changes():
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8").splitlines()


def export_html(notebook, out_dir):
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nbformat.read(notebook, as_version=4))
    (out_dir / f"{notebook.stem}.html").write_text(body, encoding="utf-8")


def missing_deliverables(report, slides):
    missing = []
    if not any(p.suffix.lower() in (".doc", ".docx") for p in report):
        missing.append("report .doc/.docx (--report)")
    if not any(p.suffix.lower() == ".pdf" for p in report):
        missing.append("report .pdf (--report)")
    if not slides:
        missing.append("slides (--slides)")
    return missing


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", required=True, help="student IDs of the team members")
    parser.add_argument("--report", nargs="*", type=Path, default=[], help="report files (.docx and .pdf)")
    parser.add_argument("--slides", nargs="*", type=Path, default=[], help="slide files (.pptx or .pdf)")
    parser.add_argument("--draft", action="store_true", help="build even if report or slides are missing")
    args = parser.parse_args()

    if not all(i.isdigit() for i in args.ids):
        parser.error("--ids must be numeric student IDs")
    for path in args.report + args.slides:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    missing = missing_deliverables(args.report, args.slides)
    if missing and not args.draft:
        parser.error("missing deliverables: " + "; ".join(missing) + " (use --draft for a test build)")

    name = "_".join(args.ids)
    out = DIST / name
    if out.exists():
        shutil.rmtree(out)

    files = source_files()
    for rel in files:
        target = out / "SourceCode" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for folder, paths in (("BaoCao", args.report), ("Slide", args.slides)):
        for path in paths:
            (out / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out / folder / path.name)
    (out / "Notebooks_HTML").mkdir(parents=True)
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        export_html(notebook, out / "Notebooks_HTML")

    status = ("> **BẢN NHÁP** — còn thiếu: " + "; ".join(missing) + "\n") if missing else ""
    guide = GUIDE_TEMPLATE.read_text(encoding="utf-8").replace("{{IDS}}", ", ".join(args.ids))
    (out / "HUONG_DAN.md").write_text(guide.replace("{{STATUS}}", status), encoding="utf-8")

    archive = DIST / f"{name}.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, Path(name) / path.relative_to(out))

    print(f"Folder : {out}")
    print(f"Zip    : {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"Source : {len(files)} files · notebooks exported to HTML: "
          f"{len(list((out / 'Notebooks_HTML').glob('*.html')))}")
    if missing:
        print("DRAFT  : missing " + "; ".join(missing))
    changes = uncommitted_changes()
    if changes:
        print(f"Note   : {len(changes)} uncommitted change(s) are included; the package differs from GitHub.")


if __name__ == "__main__":
    main()
