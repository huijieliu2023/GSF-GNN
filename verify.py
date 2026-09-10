"""Validate the local NPZ datasets required by GSF-GNN."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_KEYS = {
    "node_features",
    "node_labels",
    "edges",
    "train_masks",
    "val_masks",
    "test_masks",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate(path: Path) -> tuple[int, int]:
    with np.load(path) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"missing keys: {', '.join(sorted(missing))}")
        features, labels, edges = data["node_features"], data["node_labels"], data["edges"]
        if features.ndim != 2:
            raise ValueError("node_features must be two-dimensional")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("node_labels must be one-dimensional with one item per node")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape [num_edges, 2]")
        if len(edges) and (edges.min() < 0 or edges.max() >= len(features)):
            raise ValueError("edges contain a node index outside the feature matrix")
        for key in ("train_masks", "val_masks", "test_masks"):
            masks = data[key]
            if masks.ndim != 2 or masks.shape[1] != len(features):
                raise ValueError(f"{key} must have one column per node")
        split_count = data["train_masks"].shape[0]
        if data["val_masks"].shape[0] != split_count or data["test_masks"].shape[0] != split_count:
            raise ValueError("split masks must have the same number of rows")
    return len(features), len(edges)


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ datasets found in {DATA_DIR}")
    failures = []
    for path in paths:
        try:
            nodes, edges = validate(path)
            print(f"OK  {path.name}: {nodes} nodes, {edges} edges, sha256={digest(path)}")
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"FAIL {path.name}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(paths)} dataset archive(s).")


if __name__ == "__main__":
    main()
