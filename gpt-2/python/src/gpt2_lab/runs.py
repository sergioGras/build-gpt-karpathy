"""Prepare an experiment record without starting training or provisioning compute."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

from . import __version__
from .config import LoadedExperiment


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> dict:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).parent), *arguments],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def prepare_run(experiment: LoadedExperiment, output_dir: str | Path) -> Path:
    dataset = experiment.dataset_path
    if not dataset.is_file():
        raise ValueError(f"Dataset file does not exist: {dataset}")
    # Hash the input before making a record. The sample is tiny; later dataset
    # lessons will define provenance for collections of large token shards.
    dataset_hash = file_sha256(dataset)
    now = datetime.now(timezone.utc)
    run_id = f"{experiment.config.name}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    snapshot = asdict(experiment.config)
    # A saved configuration can be loaded from its run folder without changing
    # which local file it points to. Source templates keep portable relative paths.
    snapshot["dataset"]["path"] = str(dataset)
    snapshot_content = json.dumps(snapshot, indent=2) + "\n"
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "note": "No training has run. Training commands will be added in later lessons.",
        "created_at": now.isoformat(),
        "config_source": str(experiment.config_path),
        "config_source_sha256": experiment.config_sha256,
        "config_sha256": hashlib.sha256(snapshot_content.encode("utf-8")).hexdigest(),
        "dataset": {"path": str(dataset), "sha256": dataset_hash},
        "source": source_revision(),
        "runtime": {"package_version": __version__, "python": platform.python_version(),
                    "executable": sys.executable, "platform": platform.platform()},
    }
    (run_dir / "config.json").write_text(snapshot_content, encoding="utf-8", newline="\n")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return run_dir
