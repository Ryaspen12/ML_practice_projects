# Gait Classifier Model

## Description
- classify gait/activity
- compare classical ML vs neural nets
- evaluate engineered features

## Data Analysis

### Dataset
[Activity Monitoring Dataset](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor. 
The dataset can be used for activity recognition and intensity estimation, while developing and applying algorithms of data processing, segmentation, feature extraction and classification.
**Sensors**
3 Colibri wireless inertial measurement units (IMU):
  - sampling frequency: 100Hz
  - position of the sensors:
       - 1 IMU over the wrist on the dominant arm 
       - 1 IMU on the chest 
       - 1 IMU on the dominant side's ankle 
HR-monitor:
  - sampling frequency: ~9Hz

**Data collection protocol**
Each of the subjects had to follow a protocol, containing 12 different activities. The folder *Protocol* contains these recordings by subject.
Furthermore, some of the subjects also performed a few optional activities. The folder *Optional* contains these recordings by subject.

**Data files**
Raw sensory data can be found in space-separated text-files (.dat), 1 data file per subject per session (protocol or optional). Missing values are indicated with NaN. 
One line in the data files correspond to one timestamped and labeled instance of sensory data. 
The data files contain 54 columns: each line consists of a timestamp, an activity label (the ground truth) and 52 attributes of raw sensory data.

### Pipeline
Raw IMU → filtering → windowing → features → models

### IMU data
Trunk IMU, Dominant Wrist IMU, Dominant Ankle IMU

### Features
Using a rolling window on IMU data:
- mean
- std
- RMS
- SMA
- jerk
- dominant frequency
- spectral entropy
- autocorrelation
- peak cadence

## Models
- Logistic Regression
- Random Forest
- XGBoost
- CNN/LSTM

## Results
Accuracy/F1/AUC table.

## Future Improvements
Real-time inference, deployment, edge optimization, etc.
