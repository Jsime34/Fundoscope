# Random Forest: grouped cross-validation

Trained 2026-11-23 on 166 labeled parcels (192 parcel-campaigns, 8,715 pixels). 5-fold cross-validation grouped by parcel: each fold is scored on parcels the model never saw. Each parcel-campaign gets the class with the highest mean probability over its pixels.

**Macro-F1: 0.94** (meets the target; FR-06 requires at least 0.65, target 0.75).

| Class | Parcels | Parcel-campaigns | F1 |
|---|---:|---:|---:|
| fallow/bare | 22 | 22 | 0.98 |
| avocado | 23 | 38 | 0.96 |
| other annual crops | 88 | 88 | 0.96 |
| alfalfa | 17 | 18 | 0.91 |
| grape | 16 | 26 | 0.90 |

Confusion matrix (rows: labeled class, columns: predicted class):

| | alfalfa | avocado | fallow/bare | grape | other annual crops |
|---|---:|---:|---:|---:|---:|
| **alfalfa** | 16 | 0 | 0 | 0 | 2 |
| **avocado** | 1 | 35 | 0 | 1 | 1 |
| **fallow/bare** | 0 | 0 | 21 | 0 | 1 |
| **grape** | 0 | 0 | 0 | 22 | 4 |
| **other annual crops** | 0 | 0 | 0 | 0 | 88 |

Settings compared (500 trees, class weights `balanced_subsample`):

| Setting | Macro-F1 |
|---|---:|
| max_features=sqrt, min_samples_leaf=1 | 0.92 |
| max_features=sqrt, min_samples_leaf=5 (chosen) | 0.94 |
| max_features=0.3, min_samples_leaf=1 | 0.93 |
| max_features=0.3, min_samples_leaf=5 | 0.94 |

Notes:

- The score is the best of 4 settings, so it is slightly optimistic. The field-verified parcels are not used here: they are scored once, with the final model.
- Neighboring fields can fall in different folds. Fields next to each other tend to look alike, so a new area may score lower.
- 22 fallow/bare parcels come from satellite images (bare all campaign) instead of Street View, because idle fields next to Street View roads are rare.
- 7 labels that didn't fit their NDVI curve were reviewed by hand and kept.

## Season-to-date models (FR-14)

A campaign still in progress is classified by a model trained on its first months only, August to the latest month, with the same setting and folds. Each is scored against the crop of the whole campaign.

| Months | August to | Macro-F1 | alfalfa | avocado | fallow/bare | grape | other annual crops |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Aug | 0.67 | 0.56 | 0.81 | 0.62 | 0.54 | 0.80 |
| 2 | Sep | 0.78 | 0.71 | 0.87 | 0.91 | 0.56 | 0.87 |
| 3 | Oct | 0.82 | 0.74 | 0.93 | 0.95 | 0.59 | 0.88 |
| 4 | Nov | 0.85 | 0.80 | 0.93 | 0.95 | 0.67 | 0.90 |
| 5 | Dec | 0.87 | 0.82 | 0.93 | 0.95 | 0.71 | 0.92 |
| 6 | Jan | 0.89 | 0.86 | 0.93 | 0.95 | 0.77 | 0.93 |
| 7 | Feb | 0.90 | 0.89 | 0.93 | 0.95 | 0.77 | 0.94 |
| 8 | Mar | 0.91 | 0.89 | 0.93 | 0.98 | 0.80 | 0.94 |
| 9 | Apr | 0.91 | 0.86 | 0.96 | 0.98 | 0.80 | 0.94 |
| 10 | May | 0.94 | 0.89 | 0.97 | 0.98 | 0.88 | 0.96 |
| 11 | Jun | 0.94 | 0.89 | 0.97 | 0.98 | 0.88 | 0.96 |
| 12 | Jul | 0.94 | 0.91 | 0.96 | 0.98 | 0.90 | 0.96 |

The last row is the full model. A month in progress is classified with the model of its length, using the clear images so far.
