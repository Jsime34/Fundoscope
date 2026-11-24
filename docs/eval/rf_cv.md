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
