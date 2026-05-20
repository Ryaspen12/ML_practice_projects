# ML_practice_projects
repo for hands on practice in classical ML and potentially some neural networks

## Project 1 - Gait Classification
**Data**
- synthetic wearable data
- public datasets

**Features to engineer**
- mean/std
- spectral entropy
- peak frequency
- cadence
- jerk
- signal magnitude area
- autocorrelation
- symmetry metrics

**Models**
- Random Forest / XGBoost baseline
- simple 1D CNN or LSTM extension

## Project 2 - Healthy vs Injured Classification
**Data**
- synthetic wearable data
- public datasets
- workout + sleep metrics

**Focus**
- class imbalance
- feature importance
- interpretability
- threshold tuning

**Models**
- Logistic Regression
- Random Forest
- XGBoost
- shallow NN

## Project 3 — Edge ML Demo
tbd
Examples:
classify motion from accelerometer
convert model to ONNX / TorchScript
inference simulation

## Folder Structure
project/
│
├── data/
    ├── raw/
    ├── intermediate/
    ├── results/
├── notebooks/
├── src/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
