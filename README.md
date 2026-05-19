# Cricket Match Outcome Predictor (Random Forest)

A machine learning project that predicts the outcome of cricket matches using a Random Forest classifier. The dataset is sourced from a comprehensive collection of 15,000+ cricket match JSON files and preprocessed into CSV format for model training and evaluation.

## Table of Contents
* [Project Overview](#project-overview)
* [Data Source](#data-source)
* [Installation](#installation)
* [Usage](#usage)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
   * [Confusion Matrix](#confusion-matrix)
* [License](#license)

## Project Overview

This repository contains code to train and evaluate a Random Forest classifier on cricket match data. The goal is to predict the match winner based on match features such as teams, venue, city, toss winner, and toss decision. The model achieves good accuracy in predicting match outcomes and can be used to forecast results of upcoming cricket matches.

## Data Source

The dataset consists of over 15,000 cricket matches in JSON format, converted to CSV for easier processing. Each match record includes:

- Match date and venue information
- Teams playing
- Toss winner and decision
- Match outcome
- Additional engineered features such as:
  - Team performance metrics
  - Venue statistics
  - Toss impact factors

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cricket-rf-predictor.git
cd cricket-rf-predictor
```

2. Create a virtual environment and install dependencies:

```bash
virtualenv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. Run the raw data parsing script to convert json data:

```bash
python parse_data.py
```

2. Process the data and compute stats:

```bash
python process_data.py
```

3. Run the model training script:

```bash
python create_cricket_model.py
```

4. To make predictions for future matches:

```bash
python predict.py
```

## Model Training

The model uses scikit-learn's `RandomForestClassifier` with the following hyperparameters:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
```

The pipeline includes:
- Preprocessing steps for both categorical and numerical features
- Feature engineering to capture cricket-specific knowledge
- Data splitting into training and testing sets
- Model training with optimized hyperparameters

## Evaluation

The model is evaluated using standard classification metrics:

- Accuracy: The proportion of correctly predicted outcomes
- Precision: The ratio of correct positive predictions to all positive predictions
- Recall: The ratio of correct positive predictions to all actual positives
- F1-score: The harmonic mean of precision and recall

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix visualizes the model's performance as a binary classifier, comparing the true match outcomes against its predicted outcomes (Team 1 vs Team 2).

## Example Prediction Code

```python
from predict import make_prediction

prediction, probabilities = make_prediction(
    team1="India",
    team2="Australia",
    venue="Narendra Modi Stadium",
    city="Ahmedabad",
    toss_winner="Australia",
    toss_decision="field",
    team1_form=0.90,
    team2_form=0.80,
    head_to_head=0.45,
    match_date="2023-11-19"
)

print(f"Predicted winner: {prediction}")
```
