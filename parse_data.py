import os
import pandas as pd
import ujson as json  # Faster JSON parsing
from multiprocessing import Pool, cpu_count, freeze_support

def parse_match_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        info = data.get("info", {})
        return {
            "match_date": info.get("dates", [None])[0],
            "venue": info.get("venue"),
            "city": info.get("city"),
            "team_1": info.get("teams", [None, None])[0],
            "team_2": info.get("teams", [None, None])[1],
            "toss_winner": info.get("toss", {}).get("winner"),
            "toss_decision": info.get("toss", {}).get("decision"),
            "match_winner": info.get("outcome", {}).get("winner")
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_all_matches(directory):
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    
    with Pool(processes=cpu_count()) as pool:
        match_data = pool.map(parse_match_data, file_paths)
    
    return pd.DataFrame([match for match in match_data if match is not None])

if __name__ == "__main__":
    freeze_support()  # Required for Windows multiprocessing

    # Set your directory path containing JSON files
    directory_path = "C:/Users/pnith/OneDrive/Desktop/Cricket/all_male_json"

    # Process all JSON files and create a DataFrame
    matches_df = process_all_matches(directory_path)

    # Save to CSV
    matches_df.to_csv("cricket_matches.csv", index=False)

    print("Data extraction complete! First few rows:")
    print(matches_df.head())
