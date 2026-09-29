# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the pipeline regression fixture, at the commit that produced Phase D run 10.

    python tests/make_pipeline_fixtures.py

Runs every step of ``pipeline_runner`` on the small fixed dataset and records, per step, the exit
code and a digest of its normalised output; figure steps also record which files they wrote.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline_runner as pr  # noqa: E402


def collect(workdir: Path) -> dict[str, object]:
    """Run the whole pipeline in ``workdir`` and return what every step produced."""
    record: dict[str, object] = {"data": pr.build_dataset(workdir), "reports": {}, "figures": {}}
    reports: dict[str, dict[str, object]] = {}
    for name, args in pr.report_steps(workdir).items():
        done = pr.run_step(args, workdir)
        text = pr.normalise_text(done.stdout, workdir)
        reports[name] = {"returncode": done.returncode,
                         "stdout_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "stdout_head": text[:400]}
    reports["verify_phaseD_json"] = {
        "sha256": pr.normalised_file_digest(workdir / "phaseD_checks.json", workdir)}
    reports["constraint_summary"] = {
        "sha256": pr.normalised_file_digest(workdir / "constraint_check" / "constraint_summary.csv",
                                            workdir)}
    record["reports"] = reports
    figures: dict[str, dict[str, object]] = {}
    for name, args in pr.figure_steps(workdir).items():
        before = {p.name for p in (workdir / "figures").rglob("*") if p.is_file()} \
            if (workdir / "figures").exists() else set()
        done = pr.run_step(args, workdir)
        after = {p.name for p in (workdir / "figures").rglob("*") if p.is_file()}
        figures[name] = {"returncode": done.returncode, "files": sorted(after - before)}
    record["figures"] = figures
    return record


def main() -> None:
    """Write ``tests/fixtures/pipeline_run10.json``."""
    with tempfile.TemporaryDirectory() as tmp:
        record = collect(Path(tmp))
    record["generated_at_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=pr.REPO_ROOT,
        check=False).stdout.strip()
    out = HERE / "fixtures" / "pipeline_run10.json"
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    failed = [k for k, v in {**record["reports"], **record["figures"]}.items()
              if isinstance(v, dict) and v.get("returncode")]
    print(f"{len(record['data'])} data files, {len(record['reports'])} reports, "
          f"{len(record['figures'])} figure steps; non-zero exit: {failed} -> {out}")


if __name__ == "__main__":
    main()
