# data_prep — one-time dataset preprocessing (the WRITE side)

Tooling to convert a raw dataset into the standardized **canonical** form that the
runtime reads (`data/mmiu/datasets/<task>.json`). It is NOT imported by the
runtime package (`mosaic_gym.*`); the runtime only *reads* the already-canonical
files via `mosaic_gym.runner.data`.

```
raw MMIU (data/mmiu/all.parquet + image dirs)
   │  prepare_mmiu_canonical.py          ← convert + emit canonical
   ▼
data/mmiu/datasets/<task>.json           ← what the runtime runs on
   │  mosaic_gym.runner.data             ← (READ side, in the runtime package)
   ▼
run_episode(...)
```

- **`prepare_mmiu_canonical.py`** — the MMIU converter: reads `all.parquet` + the
  extracted image category dirs and writes one canonical `<task>.json` per MMIU
  subset. Run it:
  ```
  python scripts/data_prep/prepare_mmiu_canonical.py --mmiu-root data/mmiu
  ```
- `canonical.py` — the canonical sample schema (`REQUIRED_FIELDS`) + `build_canonical_sample`.
- `validators.py` — `validate_canonical_sample` (schema QA).
- `adapters.py` — `DatasetAdapter` base class; subclass it to add a NEW source dataset.

The canonical shape (`sample_id`, `dataset_name`, `prompt`, `images:[{image_id,
uri_or_path}]`, `task_type`, `target`) is the contract between this WRITE side and
the runtime READ side (`mosaic_gym.runner.data`). Run from this directory so the
sibling imports resolve.
