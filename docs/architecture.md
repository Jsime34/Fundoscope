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
| `ingest` | `utm.py`, `readers.py`, `parcels.py` | `utm_epsg`, `to_utm`, `from_utm`, `read_features`, `parse_parcel`, `Parcel`, `ParcelFeature`, `ParcelError` | **done** |
| `sentinel` | `cdse.py`, `evalscripts/`, `statistical.py`, `composites.py`, `quota.py` | daily stats, monthly composites | started (`cdse.py`) |
| `timeseries` | `campaigns.py`, `clean.py`, `metrics.py`, `active_area.py` | `campaign_of`, cleaning, campaign metrics | planned (week 4) |
| `ml` | `features.py`, `train_rf.py`, `train_transformer.py`, `predict.py`, `evaluate.py` | features, training, prediction | planned (weeks 5–6, 11) |
| `analysis` | `flags.py`, `neighbors.py`, `water.py`, `report.py` | flags, neighbor band, water check, report | planned (week 7) |
| `valley` | `classify.py`, `areas.py`, `render.py` | pixel classes, hectares with CI, map image | planned (week 10) |
| `db` | `models.py`, `session.py`, `repositories.py` | tables and queries | planned (week 3) |
| `pipeline` | `analyze_parcel.py`, `refresh_valley.py` | end-to-end jobs | planned (weeks 3–10) |
| `api` | `main.py`, `routes/`, `schemas.py`, `worker.py` | REST endpoints | started (`/health`) |
| `cli` | `cli.py` | `fundoscope` command | started (`check-db`, `check-cdse`) |

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
