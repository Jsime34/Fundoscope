# Notebooks

Exploration and learning notebooks, numbered in order (`01_parcel_crs.ipynb`, `02_sentinel_ndvi.ipynb`, …).
Code that works here gets moved into `backend/src/fundoscope/` with tests.

**This repository is public.** Outputs are stripped automatically on commit by `nbstripout`
(run `uv run --project ../backend nbstripout --install` once per clone), but still:

- never load files from `data/private/` in a notebook you plan to commit;
- never print credentials or partner names, coordinates, or maps.

Use the backend environment as the Jupyter kernel (`backend/.venv`).
