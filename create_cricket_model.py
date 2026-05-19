import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(file_path):
    # First load without parsing dates to see raw values
    df = pd.read_csv(file_path)
    
    # Print first few date values and their dtype
    print("\nDate format check:")
    print("First 5 raw date values:")
    print(df['match_date'].head())
    print("\nData type:", df['match_date'].dtype)
    
    # Now try to parse dates with error reporting
    try:
        df['match_date'] = pd.to_datetime(df['match_date'])
        print("\nDate conversion successful")
        print("Sample converted dates:")
        print(df['match_date'].head())
    except Exception as e:
        print("\nError converting dates:", str(e))
        # Print problematic values
        print("\nProblematic date values:")
        mask = pd.to_datetime(df['match_date'], errors='coerce').isna()
        print(df.loc[mask, 'match_date'])
    
    print(f"\nDataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Data preprocessing
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Convert date to datetime
    data['match_date'] = pd.to_datetime(data['match_date'])
    
    # Extract date features
    data['year'] = data['match_date'].dt.year
    data['month'] = data['match_date'].dt.month
    
    # Fill missing values
    data['city'] = data['city'].fillna('Unknown')
    
    # Ensure all string columns are treated as categorical
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
    
    # Drop rows with missing values in important columns
    data = data.dropna(subset=['match_winner'])
    
    # Target variable
    y = data['match_winner']
    
    # Features to use for prediction
    X = data.drop(['match_winner', 'match_date'], axis=1)
    
    # Identify categorical columns (now includes all object and category dtype columns)
    categorical_cols = [col for col in X.columns if X[col].dtype.name == 'category']
    
    # Identify numerical columns
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    return X, y, categorical_cols, numerical_cols

# Feature engineering - simplified
def engineer_features(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure match_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['match_date']):
        data['match_date'] = pd.to_datetime(data['match_date'], errors='coerce')
    
    # Drop rows where date conversion failed
    data = data.dropna(subset=['match_date'])
    
    # Create simple derivative features
    data['toss_bat'] = (data['toss_decision'] == 'bat').astype(int)
    
    # Add month as a cyclical feature
    data['month_sin'] = np.sin(2 * np.pi * data['match_date'].dt.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['match_date'].dt.month / 12)
    
    # Toss advantage: did toss winner also win the match?
    data['toss_winner_won'] = (data['toss_winner'] == data['match_winner']).astype(int)
    
    return data

# Build preprocessing pipeline
def build_pipeline(categorical_cols, numerical_cols):
    # Create preprocessing steps for different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor

# Train the model with more robust approach
def train_model(X, y, preprocessor):
    # Filter out rare classes (teams that appear less than 2 times)
    value_counts = y.value_counts()
    valid_classes = value_counts[value_counts >= 2].index
    mask = y.isin(valid_classes)
    
    # Filter X and y
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Original number of samples: {len(y)}")
    print(f"Number of samples after filtering rare classes: {len(y_filtered)}")
    print(f"Number of classes after filtering: {len(valid_classes)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_filtered
    )
    
    # Count unique classes for stratification
    class_counts = y_filtered.value_counts()
    min_classes = class_counts[class_counts > 2].count()
    
    # Use fewer CV splits if we have classes with few samples
    cv_splits = 3 if min_classes >= 3 else 2
    
    # Create CV strategy with stratification
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # Create pipeline for Random Forest (simpler approach)
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, 
                                             class_weight='balanced'))
    ])
    
    # Start with a simpler approach without GridSearchCV
    print("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Create and save confusion matrix for top classes
    plot_classes = min(10, len(np.unique(y_filtered)))  # Limit to top classes if many
    top_classes = y_filtered.value_counts().nlargest(plot_classes).index.tolist()
    
    # Filter test data for top classes
    mask = y_test.isin(top_classes)
    if mask.sum() > 0:  # Only if we have examples
        y_test_filtered = y_test[mask]
        y_pred_filtered = [pred if pred in top_classes else 'Other' for pred in y_pred[mask]]
        
        # Create confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=top_classes, 
                    yticklabels=top_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Top Classes)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return rf_pipeline

# Function to make predictions for future matches
def predict_match(model, match_data):
    """
    Predict the outcome of a cricket match
    
    Parameters:
    model: trained model
    match_data: DataFrame with a single row containing match details
    
    Returns:
    predicted_winner: string with team name
    probabilities: dictionary with team probabilities
    """
    # Make prediction
    prediction = model.predict(match_data)
    probabilities = model.predict_proba(match_data)
    
    # Get team probabilities
    teams = model.classes_
    team_probs = {}
    for i, team in enumerate(teams):
        team_probs[team] = probabilities[0][i]
    
    # Sort probabilities
    sorted_probs = {k: v for k, v in sorted(team_probs.items(), key=lambda item: item[1], reverse=True)}
    
    return prediction[0], sorted_probs

# Main function
def main():
    # File path
    file_path = 'cricket_matches_enhanced.csv'  # Update with your CSV file path
    
    # Load data
    df = load_data(file_path)
    
    # Feature engineering - simplified
    df_engineered = engineer_features(df)
    
    # Preprocess data
    X, y, categorical_cols, numerical_cols = preprocess_data(df_engineered)
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Add before train_model call:
    print("\nClass distribution:")
    print(y.value_counts())
    print("\nTeams with only one match:")
    print(y.value_counts()[y.value_counts() == 1])
    
    # Build preprocessing pipeline
    preprocessor = build_pipeline(categorical_cols, numerical_cols)
    
    # Train and evaluate model
    model = train_model(X, y, preprocessor)
    
    # Save the model
    import joblib
    joblib.dump(model, 'cricket_prediction_model.joblib')
    print("Model saved as 'cricket_prediction_model.joblib'")
    
    # Example of how to create data for prediction with correct column names
    new_match_data = pd.DataFrame({
        'team_1': ['India'],
        'team_2': ['Sri Lanka'],
        'venue': ['R. Premadasa Stadium'],  # Add venue
        'city': ['Colombo'],
        'toss_winner': ['Sri Lanka'],
        'toss_decision': ['bat'],
        'team_1_recent_form': [0.6],  # Add team form metrics
        'team_2_recent_form': [0.5],
        'team_1_head_to_head': [0.55],
        'year': [2023],
        'month': [10],
        'toss_bat': [1],
        'month_sin': [np.sin(2 * np.pi * 10 / 12)],
        'month_cos': [np.cos(2 * np.pi * 10 / 12)],
        'toss_winner_won': [1]
    })
    
    # Print the columns required by the model
    print("\nRequired columns for prediction:")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    predicted_winner, probabilities = predict_match(model, new_match_data)
    print(f"\nPredicted Winner: {predicted_winner}")
    print(f"Probabilities: {probabilities}")
    
if __name__ == "__main__":
    main()