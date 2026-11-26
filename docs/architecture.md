# Architecture

Fundoscope is built one module at a time, and inside each module one **submodule** (one `.py` file with a single responsibility) at a time. Every submodule is finished (tested, linted, documented) before the next one starts.

## Layers

```
interfaces      api, cli                 HTTP and command line. No business logic.
orchestration   pipeline                 The only place that combines domain logic with I/O.
domain          analysis, valley         Flags, reports, valley hectares.
                ml                       Crop models.
                timeseries, ingest       Data preparation.
adapters        sentinel, db             Talk to Copernicus and PostGIS.
base            config                   Settings.
```

Rules, checked automatically by `uv run lint-imports` (contracts in `backend/pyproject.toml`):

1. **Imports only go down.** A module may import from layers below it, never above. Modules on the same line are independent of each other.
2. **Domain modules are pure.** `ingest`, `timeseries`, `ml`, `analysis`, and `valley` never touch the database, the network, or the web framework. Data goes in, data comes out. That keeps their tests fast and deterministic, and lets the CLI, API, and notebooks reuse them.
3. **`pipeline` does the wiring.** For example, "analyze a parcel" = read the cache from `db`, fetch the missing dates from `sentinel`, compute with the domain modules, write back to `db`.
4. **Use the public API.** Each module exports what others may use in its `__init__.py` (`__all__`). Import from the package (`from fundoscope.ingest import parse_parcel`), not from its internal files.

## Module map

| Module | Submodules | Public API | Status |
|---|---|---|---|
| `ingest` | `utm.py`, `readers.py`, `parcels.py`, `labels.py` | `utm_epsg`, `to_utm`, `from_utm`, `read_features`, `parse_parcel`, `Parcel`, `ParcelFeature`, `ParcelError`, `read_labels`, `check_labels`, `LabelReport`, `LabelError`, `CROP_CLASSES` | **done** |
| `sentinel` | `cdse.py`, `evalscripts.py`, `statistical.py`, `composites.py`, `landcover.py`, `quota.py` | `sh_config`, `fetch_daily_stats`, `DailyStats`, `fetch_monthly_composites`, `MonthlyComposites`, `fetch_land_cover` (ESA WorldCover), `batch_cap`, `CreditBudget` | **done** |
| `timeseries` | `campaigns.py`, `indices.py`, `clean.py`, `metrics.py`, `active_area.py` | `campaign_of`, `ndvi`, `ndwi`, `clean_series`, `campaign_metrics`, `active_fraction`, `max_ndvi_composite` | **done** |
| `ml` | `features.py`, `label_curves.py`, `train_rf.py`, `train_transformer.py`, `predict.py`, `evaluate.py` | `pixel_features`, `feature_columns`, `season_to_date`, `training_sample`, `FEATURE_COLUMNS`, `FeatureError`, `check_curves`, `CurveCheck`, `training_table`, `cross_validate`, `fit`, `score`, `predict_crop`, `CropPrediction` | in progress (all but `train_transformer` done) |
| `analysis` | `flags.py`, `neighbors.py`, `water.py`, `report.py` | `campaign_flags`, `Flag` (abandonment, crop change, anomaly, low data); `neighbor_ring`, `compare`, `NeighborBand`, `underperforming_flags` (FR-08); water check, report | in progress (`flags` and `neighbors` done) |
| `valley` | `aoi.py`, `season.py`, `farmland.py`, `classify.py`, `areas.py`, `render.py` | `valley` (tiles), `season` (what a refresh counts), `farmland_mask`, `classify_tile` (with the "planted, not yet identifiable" rule), `class_areas` (hectares with 95 % ranges, Olofsson 2014), `render` (one transparent PNG per class, fixed colors, planted striped) | **done** |
| `db` | `records.py`, `models.py`, `parcels.py`, `memory.py`, `session.py`, `migrations/` | `ParcelStore` (protocol), `SqlParcelStore` (PostGIS), `MemoryParcelStore`, `StoredParcel`, `StoredHistory`, `StoredCampaign`, `session_scope`, `upgrade`, `describe`; tables `parcels` (GiST on `geom`), `observations`, `campaigns`, `jobs` | **done** (`jobs` is used by FR-13) |
| `pipeline` | `parcel_history.py`, `training_data.py`, `train_model.py`, `parcel_store.py`, `valley_store.py`, `valley_refresh.py`, `neighbors.py` | `parcel_history`, `history_of` (from stored data), `analyze_parcel`, `reopen_parcel`, `parcel_key`, `ParcelHistory`, `extract_training_data`, `extraction_tasks`, `load_training_pixels`, `train_crop_model` (full and season-to-date models), `load_crop_model`, `model_confusion`, `interior_stack`, `load_valley`, `plan_refresh`, `refresh_valley`, `neighbor_report`; end-to-end jobs | in progress (`parcel_history` with crop prediction and neighbors, `training_data`, `train_model`, `valley_refresh`, `neighbors`, and `parcel_store` done; background jobs next) |
| `api` | `main.py`, `history.py`, `valley.py`, `schemas.py`, `worker.py` | `POST /api/history`, `GET /api/parcels`, `GET /api/parcels/{id}/history`, `GET /api/valleys/{slug}`, `GET /api/valleys/{slug}/maps/{count}/{image}.png`, `/health` | in progress (no background jobs yet) |
| `cli` | `checks.py`, `db.py`, `history.py`, `plots.py`, `labels.py`, `model.py`, `valley.py` | `fundoscope check-db`, `check-cdse`, `db upgrade`, `db tables`, `history <file>`, `labels check`, `labels extract`, `model train`, `valley refresh <slug>` | in progress |

## Definition of done (every submodule)

- [ ] Public functions and classes have type hints and a docstring stating inputs, outputs, **units** (m, ha), and **CRS**.
- [ ] Respects its layer (pure if it's a domain module); no hidden global state.
- [ ] Unit tests in `backend/tests/<module>/test_<submodule>.py`: normal case, edge cases, and the error messages users will see.
- [ ] No network or database in unit tests (use small fixtures or recorded responses).
- [ ] All checks pass (below).
- [ ] Exported in the module's `__init__.py` `__all__`.
- [ ] One focused commit, e.g. `feat(ingest): read KML/KMZ/GeoJSON uploads`.
- [ ] Jaime can explain how it works (demo Q&A).

## Workflow per submodule

1. Agree on the interface: signatures and docstrings.
2. Write the tests.
3. Implement until the tests pass.
4. Run the checks.
5. Review, then commit.

Checks, from `backend/`:

```
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run pytest --cov=fundoscope --cov-report=term-missing
```

## Conventions

- **Coordinates:** WGS84 longitude/latitude, always in x=lon, y=lat order (`always_xy=True` in pyproj).
- **Measurements:** in UTM (EPSG:32717 for Piura/La Libertad, EPSG:32718 for Ica), meters and hectares; EPSG codes are `int`.
- **Geometries:** Shapely 2.
- **Campaigns:** August 1 – July 31, written as `"2023/24"`.
- **Errors:** each module raises its own exception (subclass of `ValueError`) with a message a user can act on; the API turns these into HTTP 422.
- **Commits:** Conventional Commits (`feat`, `fix`, `test`, `docs`, `refactor`, `chore`) with the module as scope.
