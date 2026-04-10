# Chicago Traffic Crash Severity Prediction

**CMU ML Foundations — Midterm Project (Group 10)**
Diyouva Novith, Rizaldy Utomo, Utami

## Overview

A data-driven framework for proactive safety resource allocation in Chicago. We predict severe crash outcomes (fatal/incapacitating injuries) using 328,495 crash records from 2023-2025, combining supervised classification with condition-level risk scoring and spatial clustering.

**Two modeling tracks:**
- **Track A (Crash-Level Classification):** Logistic Regression, Random Forest, XGBoost on 98 engineered features. Best: Random Forest (AUC-ROC 0.8428).
- **Track B (Condition-Level Risk):** Aggregates crashes into 91,616 location-condition profiles with empirical-Bayes risk scoring. Produces a ranked priority list for CDOT deployment.

## Data

**Source:** [Traffic Crashes - Crashes | Chicago Data Portal](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if)

Download the CSV and place it in `dataset/` as `training_data_2023_2025.csv`, then run notebooks in order.

## Project Structure

```
notebooks/
  01_data_preparation.ipynb    Data cleaning, target creation, temporal split
  02_eda.ipynb                 Feature engineering + exploratory data analysis
  03_classification.ipynb      Track A (crash-level) + Track B (condition-level risk)
  04_clustering.ipynb          K-Means and DBSCAN spatial clustering
  05_spatial_dashboard.ipynb   Interactive HTML dashboard + policy decision rule

dataset/
  cleaned_data/
    train.csv                  Base cleaned training split (246,221 rows)
    test.csv                   Base cleaned test split (82,274 rows)
    train_model.csv            Engineered features for modeling (98 features)
    test_model.csv             Engineered features for modeling
    feature_columns.json       Feature contract (column list + metadata)
    risk_scores.csv            Condition-profile risk rankings (91,616 profiles)
    data_prep_metadata.json    Split metadata

image/
  02_eda/                      EDA figures
  03_classification/           Model evaluation plots
  05_dashboard/                Dashboard assets + HTML

paper/
  report.md                    Full project report

plan/                          Execution plans for each phase
```

## Pipeline

```
training_data_2023_2025.csv
        |
  01_data_preparation.ipynb
        |
  train.csv, test.csv
        |
  02_eda.ipynb
        |
  train_model.csv, test_model.csv, feature_columns.json
        |
  03_classification.ipynb
        |
  risk_scores.csv ──────> 04_clustering.ipynb ──> cluster_labels.csv
        |                                                |
        └────────────────────────────────────────────────┘
                                |
                   05_spatial_dashboard.ipynb
```

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```
2. Place raw data in `dataset/training_data_2023_2025.csv`
3. Run notebooks in order: `01` → `02` → `03` → `04` → `05`

## Key Findings

- Severe crashes are rare (1.61%) but predictable from environmental and roadway features
- Pedestrian crashes carry a 15.4% severe rate — 10x the city average
- Late-night and weekend hours have 2-4x the severity rate of daytime
- Darkness on lighted roads under clear skies is the dominant high-risk condition
- Top 50 priority condition profiles provide actionable targets for CDOT resource deployment
