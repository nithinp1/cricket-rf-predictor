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
* [Contributing](#contributing)
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
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. Ensure the processed CSV data is placed in the `data/` directory.
2. Run the model training script:

```bash
python train_model.py --data_path data/cricket_matches.csv --output_dir models/
```

3. To make predictions for future matches:

```bash
python predict_match.py --model_path models/cricket_prediction_model.joblib --match_data examples/new_match.csv
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

![Confusion Matrix](results/confusion_matrix.png)

The confusion matrix visualizes the model's performance across different teams, showing how well it predicts each team's wins.

## Example Prediction Code

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/cricket_prediction_model.joblib')

# Create data for a new match
new_match = pd.DataFrame({
    'venue': ['Lords Cricket Ground'],
    'city': ['London'],
    'toss_winner': ['England'],
    'toss_decision': ['bat'],
    'year': [2025],
    'month': [6],
    'toss_bat': [1],
    'month_sin': [0.5],
    'month_cos': [0.866],
    'toss_winner_won': [0]
})

# Make prediction
prediction = model.predict(new_match)
probabilities = model.predict_proba(new_match)

print(f"Predicted winner: {prediction[0]}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
