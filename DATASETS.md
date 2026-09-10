# Bundled datasets

The following archives are stored in `data/` and are loaded locally; the training program never downloads data.

| Dataset basename | Archive filename |
| --- | --- |
| Actor | `actor.npz` |
| Amazon Ratings | `amazon_ratings.npz` |
| Chameleon | `chameleon.npz`, `chameleon_directed.npz`, `chameleon_filtered.npz`, `chameleon_filtered_directed.npz` |
| Citeseer | `citeseer.npz` |
| Cora | `cora.npz` |
| Cornell | `cornell.npz` |
| Minesweeper | `minesweeper.npz` |
| PubMed | `pubmed.npz` |
| Questions | `questions.npz` |
| Roman Empire | `roman_empire.npz` |
| Squirrel | `squirrel.npz`, `squirrel_directed.npz`, `squirrel_filtered.npz`, `squirrel_filtered_directed.npz` |
| Texas | `texas.npz`, `texas_4_classes.npz` |
| Tolokers | `tolokers.npz` |
| Wisconsin | `wisconsin.npz` |

Every archive must provide the following arrays:

- `node_features`: two-dimensional float-compatible node-feature matrix.
- `node_labels`: one label per node.
- `edges`: shape `[num_edges, 2]`, containing zero-based source and destination node indices.
- `train_masks`, `val_masks`, `test_masks`: Boolean split matrices with one row per split and one column per node.

Run `python verify.py` after cloning or moving the data to validate structure and print a SHA-256 digest for each archive. The digest makes a local copy identifiable; it does not establish provenance or licensing.

These files were migrated from the supplied research workspace. Before public redistribution, obtain and follow the original terms for every dataset. In particular, do not infer a dataset license from this repository’s future code license.
