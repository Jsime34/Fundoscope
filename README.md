# Fundoscope
Fundoscope will turn usable Copernicus satellite data into: A live crop count for a valley: the hectares of each crop in the ground this campaign compared with past campaigns. It will be updated weekly. And, a multi-year history for any single parcel, so a fundo can be screened before anyone pays for a detailed appraisal.

> **Status:** senior project in progress (September – December 2026). Pilot regions: Ica, La Libertad, and Piura, Peru.

## What it will do

- **Valley crop monitor:** classifies every farmed pixel in a pilot valley from Sentinel-2 imagery and reports the hectares of each crop for the current campaign (August–July) against previous ones, with error ranges. Refreshed weekly. Mobile-friendly, with Spanish labels.
- **Parcel report:** upload a parcel boundary (KML or GeoJSON) and get its history since 2018: NDVI/NDWI per campaign, % of the area actively farmed, likely crop per campaign, abandonment and crop-change flags, a comparison with neighboring farmland, and a check against aquifers under a groundwater ban.

## Stack

Python 3.12 · FastAPI · PostgreSQL + PostGIS · GeoPandas / Shapely / pyproj · Copernicus Data Space Ecosystem (Sentinel Hub Statistical API, openEO) · scikit-learn / PyTorch · React + Vite + MapLibre GL JS · Docker · Google Cloud Run

## Repository layout

```
backend/            Python package (src/fundoscope) and tests
  src/fundoscope/
    ingest/         parcel upload and validation
    sentinel/       Copernicus data access
    timeseries/     cleaning, campaigns, metrics
    ml/             crop classification
    analysis/       flags, neighbors, water check, report
    valley/         valley crop monitor
    pipeline/       end-to-end jobs that combine the modules above with I/O
    api/            REST API and job worker
    db/             models and migrations
frontend/           React app (coming in week 8)
notebooks/          exploration notebooks (outputs stripped)
docs/               requirements, architecture, evaluation
docker-compose.yml  local PostGIS
```

Module boundaries, dependency rules, and the per-module checklist are in [docs/architecture.md](docs/architecture.md). The dependency rules are enforced in CI with `lint-imports`.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/), Docker, Git.

1. Copy `.env.example` to `.env` and add your Copernicus Data Space OAuth client ID and secret.
2. Start the database (host port 5433):
   ```
   docker compose up -d db
   ```
3. Install and check the backend:
   ```
   cd backend
   uv sync
   uv run nbstripout --install   # once per clone: strips notebook outputs on commit
   uv run fundoscope check-db
   uv run fundoscope check-cdse
   uv run pytest
   uv run lint-imports
   ```
4. Run the API (http://localhost:8020/docs):
   ```
   uv run uvicorn fundoscope.api.main:app --reload --port 8020
   ```

## Data sources and attribution

- Contains modified Copernicus Sentinel data [2018–2026], processed through the Copernicus Data Space Ecosystem.
- ESA WorldCover 10 m 2021 v200 (CC BY 4.0).
- Aquifer restrictions: Autoridad Nacional del Agua (ANA), R.J. N° 330-2011-ANA and R.J. N° 0076-2025-ANA.
- Field-verified parcels are private and are not included in this repository.

## Author

Jaime Gutierrez, BYU-Idaho CSE 499 Senior Project.
