import requests
import pandas as pd

# Fetch data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract relevant data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

# Calculate statistics
players['form'] = players['form'].astype(float)
players['recent_points_per_game'] = players['total_points'] / players['minutes'] * 90
teams['fixture_difficulty'] = teams['strength_attack_home'] + teams['strength_attack_away']
players = players.merge(teams[['id', 'fixture_difficulty']], left_on='team', right_on='id', how='left')
players['expected_minutes'] = players['minutes'] / players['news'].apply(lambda x: 1 if x == '' else 2)

# Define weights
weights = {
    'form': 0.5,
    'recent_points_per_game': 0.3,
    'fixture_difficulty': -0.2,
    'expected_minutes': 0.2
}

# Calculate projected points
players['projected_points'] = (
    weights['form'] * players['form'] +
    weights['recent_points_per_game'] * players['recent_points_per_game'] +
    weights['fixture_difficulty'] * players['fixture_difficulty'] +
    weights['expected_minutes'] * players['expected_minutes']
)

# Display top recommendations
top_players = players.sort_values(by='projected_points', ascending=False)[
    ['first_name', 'second_name', 'team', 'projected_points']
].head(10)

print(top_players)
