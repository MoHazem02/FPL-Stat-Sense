import requests
import pandas as pd

# Fetch data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract relevant data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

# Preprocess data
def preprocess_data(players, teams):
    """Preprocess player and team data, calculate basic statistics."""
    players['form'] = players['form'].astype(float)
    players['recent_points_per_game'] = players['total_points'] / players['minutes'] * 90
    teams['fixture_difficulty'] = teams['strength_attack_home'] + teams['strength_attack_away']
    players = players.merge(teams[['id', 'fixture_difficulty']], left_on='team', right_on='id', how='left')
    players['expected_minutes'] = players['minutes'] / players['news'].apply(lambda x: 1 if x == '' else 2)
    return players

players = preprocess_data(players, teams)

# Helper Functions for Criteria Calculations
def calculate_recent_form(player):
    """Calculate the average points of the player in the last 4 games."""
    # Placeholder for actual implementation (requires historical game data)
    return 0

def calculate_fixture_difficulty(player, teams):
    """Calculate fixture difficulty based on team ranks."""
    team_rank = player['team_rank']
    opposition_rank = player['opposition_rank']
    if team_rank <= 3:
        team_class = 1
    elif 4 <= team_rank <= 8:
        team_class = 2
    elif 9 <= team_rank <= 15:
        team_class = 3
    else:
        team_class = 4

    if opposition_rank <= 3:
        opposition_class = 1
    elif 4 <= opposition_rank <= 8:
        opposition_class = 2
    elif 9 <= opposition_rank <= 15:
        opposition_class = 3
    else:
        opposition_class = 4

    return team_class - opposition_class

def calculate_absences_opposition(player):
    """Stub for calculating absences in the opposition team."""
    return 0

def calculate_time_in_round(player):
    """Check if the game is the first game of the day."""
    # Placeholder logic: assume we have game schedule data
    first_game_flag = player['first_game_of_round']  # This would be derived from game schedule
    return -1 if first_game_flag else 0

def calculate_rest_period(player):
    """Calculate the rest period since the last game."""
    # Placeholder for rest calculation, assumes availability of schedule data
    return player['rest_days']

# Additional Criteria Suggestions
def calculate_home_advantage(player):
    """Players perform better when playing at home."""
    return 1 if player['was_home'] else 0

def calculate_team_momentum(player):
    """Higher recent team performance boosts individual player performance."""
    return player['team_form']  # Assume team_form is a rolling average of recent results

# Objective Functions
def calculate_gk_points(player):
    """Calculate projected points for a Goalkeeper."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player),
        calculate_time_in_round(player),
        calculate_rest_period(player),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_def_points(player):
    """Calculate projected points for a Defender."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player),
        calculate_time_in_round(player),
        calculate_rest_period(player),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_mid_points(player):
    """Calculate projected points for a Midfielder."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player),
        calculate_time_in_round(player),
        calculate_rest_period(player),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_fwd_points(player):
    """Calculate projected points for a Forward."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player),
        calculate_time_in_round(player),
        calculate_rest_period(player),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

# Predict Points
players['projected_points'] = 0  # Initialize projected points column

def predict_points(players):
    """Calculate projected points for all players based on position."""
    for index, player in players.iterrows():
        if player['element_type'] == 1:  # Goalkeeper
            players.at[index, 'projected_points'] = calculate_gk_points(player)
        elif player['element_type'] == 2:  # Defender
            players.at[index, 'projected_points'] = calculate_def_points(player)
        elif player['element_type'] == 3:  # Midfielder
            players.at[index, 'projected_points'] = calculate_mid_points(player)
        elif player['element_type'] == 4:  # Forward
            players.at[index, 'projected_points'] = calculate_fwd_points(player)

predict_points(players)

# Display top recommendations
top_players = players.sort_values(by='projected_points', ascending=False)[
    ['first_name', 'second_name', 'team', 'projected_points']
].head(10)

print(top_players)