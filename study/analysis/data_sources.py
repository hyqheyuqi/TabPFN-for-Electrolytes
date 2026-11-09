"""Validate retained inputs against hashes recorded before upstream cleanup."""
import hashlib
import json
from pathlib import Path


def project_directory():
    """Locate this checkout independently of its folder name."""
    return Path(__file__).resolve().parents[1]


def ampere_data_directory():
    return project_directory() / "datasets" / "ampere"


def source_records(root):
    return json.loads((project_directory() / "reference_runs/source_manifest.json").read_text())["files"]


def recorded_source_hash(root, original, retained):
    original = str(original.relative_to(root))
    retained = str(retained.relative_to(root))
    matches = [r for r in source_records(root)
               if r["original_path"] == original and r["path"] == retained]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous source provenance: {retained}")
    return matches[0]["sha256"]


def validate_retained_sources(root):
    for record in source_records(root):
        path = root / record["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Retained source hash mismatch: {path}")
