import pandas as pd
import numpy as np
import joblib

def make_prediction(team1, team2, venue, city, toss_winner, toss_decision, 
                   team1_form, team2_form, head_to_head, match_date):
    """
    Make cricket match prediction using saved model
    """
    # Load the saved model
    model = joblib.load('cricket_prediction_model.joblib')
    
    # Convert date string to month
    date = pd.to_datetime(match_date)
    month = date.month
    year = date.year
    
    # Calculate cyclical month features
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Create DataFrame with match details
    match_data = pd.DataFrame({
        'team_1': [team1],
        'team_2': [team2],
        'venue': [venue],
        'city': [city],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'team_1_recent_form': [team1_form],
        'team_2_recent_form': [team2_form],
        'team_1_head_to_head': [head_to_head],
        'year': [year],
        'month': [month],
        'toss_bat': [1 if toss_decision == 'bat' else 0],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'toss_winner_won': [1]  # This will be updated after prediction
    })
    
    # Make prediction
    prediction = model.predict(match_data)[0]
    probabilities = model.predict_proba(match_data)[0]
    
    # Get team probabilities
    teams = model.classes_
    team_probs = {team: prob for team, prob in zip(teams, probabilities)}
    
    # Sort probabilities
    sorted_probs = dict(sorted(team_probs.items(), key=lambda x: x[1], reverse=True))
    
    return prediction, sorted_probs

def main():
    # Example usage
    prediction, probabilities = make_prediction(
    team1="India",
    team2="Australia",
    venue="Narendra Modi Stadium",
    city="Ahmedabad",
    toss_winner="Australia",
    toss_decision="field",
    team1_form=0.90,  # India's win percentage before the final
    team2_form=0.80,  # Australia's win percentage before the final
    head_to_head=0.45,  # India's historical win ratio against Australia
    match_date="2023-11-19"
    )

    
    print(f"\nPredicted Winner: {prediction}")
    print("\nWin Probabilities:")
    for team, prob in probabilities.items():
        print(f"{team}: {prob:.2%}")

if __name__ == "__main__":
    main()