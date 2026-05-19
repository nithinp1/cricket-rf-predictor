import pandas as pd

# Load the extracted match data
df = pd.read_csv("cricket_matches.csv")

# Ensure match_date is in datetime format
df["match_date"] = pd.to_datetime(df["match_date"])

# Sort data by match date
df = df.sort_values(by="match_date")

# Store team match history
team_history = {}  # {team_name: [win, win, loss, ...]}
head_to_head = {}  # {(team_1, team_2): [wins_for_team1, wins_for_team2]}

# Function to calculate recent form (last 5 matches)
def get_recent_form(team):
    return team_history.get(team, [])[-5:]  # Last 5 matches

# Process each match
recent_form_list = []
head_to_head_list = []

for index, row in df.iterrows():
    team1 = row["team_1"]
    team2 = row["team_2"]
    winner = row["match_winner"]

    # Calculate recent form (win % in last 5 matches)
    team1_recent = get_recent_form(team1)
    team2_recent = get_recent_form(team2)

    team1_form = team1_recent.count("win") / len(team1_recent) if team1_recent else 0
    team2_form = team2_recent.count("win") / len(team2_recent) if team2_recent else 0

    recent_form_list.append((team1_form, team2_form))

    # Calculate head-to-head stats
    matchup = tuple(sorted([team1, team2]))  # Ensure (A, B) and (B, A) are same
    if matchup not in head_to_head:
        head_to_head[matchup] = {team1: 0, team2: 0}  # Track by name, not index

    if winner in head_to_head[matchup]:
        head_to_head[matchup][winner] += 1

    if winner == team1:
        team_history.setdefault(team1, []).append("win")
        team_history.setdefault(team2, []).append("loss")
    elif winner == team2:
        team_history.setdefault(team1, []).append("loss")
        team_history.setdefault(team2, []).append("win")
    else:  # In case of no winner recorded
        team_history.setdefault(team1, []).append("draw")
        team_history.setdefault(team2, []).append("draw")

    # Store head-to-head win ratio for team1
    total_matches = head_to_head[matchup][team1] + head_to_head[matchup][team2]
    team1_head_to_head = head_to_head[matchup][team1] / total_matches if total_matches > 0 else 0
    head_to_head_list.append(team1_head_to_head)

# Add new features to DataFrame
df["team_1_recent_form"] = [x[0] for x in recent_form_list]
df["team_2_recent_form"] = [x[1] for x in recent_form_list]
df["team_1_head_to_head"] = head_to_head_list

# Save the enhanced dataset
df.to_csv("cricket_matches_enhanced.csv", index=False)

print("Feature engineering complete! New CSV file saved.")
