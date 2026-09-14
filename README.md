# Machine Learning Practice Projects

A collection of machine learning projects focused on developing practical skills in data analysis, feature engineering, classical machine learning, deep learning, and model evaluation.

The projects emphasize applying machine learning to real-world datasets rather than focusing solely on model training. Each project explores the full workflow from data preprocessing through feature engineering, modeling, evaluation, and interpretation.

## Projects

### 1. Activity Classification

**Wearable sensor activity recognition using classical machine learning and neural networks**

Uses the PAMAP2 Physical Activity Monitoring dataset to classify physical activities from wearable IMU data.

The project compares:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost / Gradient Boosting
* CNN
* CNN + LSTM

Key techniques include:

* Sensor data preprocessing
* Signal filtering
* Windowing and segmentation
* Time-domain feature extraction
* Frequency-domain feature extraction
* Supervised classification
* Neural networks for time-series data
* Confusion matrices and model comparison

The project investigates whether learned representations from raw sensor signals provide an advantage over carefully engineered features.

**[View Activity Classification →](Project_1_Activity_Classification/README.md)**

---

## Machine Learning Workflow

Across these projects, I am focusing on building a practical understanding of the complete machine learning workflow:

```text
Raw Data
   ↓
Data Exploration
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Train / Validation / Test
   ↓
Baseline Model
   ↓
Model Development
   ↓
Hyperparameter Optimization
   ↓
Evaluation
   ↓
Error Analysis
   ↓
Interpretation
```

The goal is to understand not only how to train models, but also **why a particular modeling approach works, where it fails, and how to evaluate whether improvements are meaningful.**

## Skills & Tools

### Python

* NumPy
* pandas
* SciPy
* scikit-learn
* PyTorch
* Matplotlib

### Machine Learning

* Regression and classification
* Ensemble methods
* Random forests
* Gradient boosting
* Feature engineering
* Hyperparameter optimization
* Model evaluation

### Deep Learning

* Convolutional neural networks
* Recurrent neural networks
* LSTMs
* Time-series modeling
* Training and validation workflows

### Signal Processing

* Filtering
* Windowing
* Statistical features
* Frequency-domain analysis
* Spectral features
* Autocorrelation
* Sensor data processing

## Goals

These projects are part of an ongoing effort to strengthen practical machine learning and data science skills, with particular interest in applications involving:

* Sensor data
* Time-series analysis
* Human movement
* Robotics and autonomous systems
* Signal processing
* Applied machine learning

The emphasis is on developing models that are **well-evaluated, interpretable, and appropriate for the problem**, rather than simply maximizing a single performance metric.
