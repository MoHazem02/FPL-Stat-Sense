import requests
import pandas as pd
import plotly.express as px

# Fetch data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_API_URL = "https://fantasy.premierleague.com/api/fixtures/"

response = requests.get(FPL_API_URL)
data = response.json()
fixtures_response = requests.get(FIXTURES_API_URL)
fixtures_data = fixtures_response.json()

# Extract relevant data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

key_players_by_team = {
    0 : ['Raya', 'Saliba', 'Rice', 'Saka'],
    2 : ['kluivert', 'semenyo'],
    13 : ['Bruno Fernandes'],
    12 : ['Rodri', 'Gvardiol', 'Kevin De Bruyne', 'Halland'],
    16 : ['Armstrong'],
    14 : ['Alexander Isak', 'Antony Gordon'],
    17 : ['Solanke', 'Van de ven', 'pedro porro'],
    7 : ['McNeil'],
    15 : ['Chris Wood', 'Sells', 'murillo'],
    5 : ['Cole Palmer', 'Enzo Fernandez'],
    10 : ['Vardy', 'Bounante'],
    3 : ['Mbuemo'],
    4 : ['Welbeck', 'Mitoma'],
    19 : ['Cunha', 'Ait-Nouri'],
    18 : ['Bowen', 'Kudus'],
    1 : ['Watkins', 'Rogers'],
    8 : ['Jiminez', 'Smith-rowe'],
    11 : ['Salah', 'Van Dijk', 'MacAllister', 'Gravenberch'],
    6 : ['Guehi', 'Mateta', 'Eze'],
    9 : ['Delap', 'Sam Morsy']
}

# Get the current gameweek
def get_current_gameweek():
    try:
        current_week = 1
        for event in data['events']:
            if not event['finished']:
                current_week = event['id']
                break
        return current_week
    except requests.RequestException as e:
        print(f"Error fetching gameweek information: {e}")
        return None

# Extract current gameweek fixtures
def get_current_fixtures(current_gameweek):
    current_fixtures = [fixture for fixture in fixtures_data if fixture['event'] == current_gameweek]
    fixtures_df = pd.DataFrame(current_fixtures)
    return fixtures_df[['team_h', 'team_a', 'kickoff_time']]

# Preprocess player and team data
def preprocess_data(players, teams):
    players['form'] = players['form'].astype(float)

    # Safely handle division by zero
    players['recent_points_per_game'] = players['total_points'] / (players['minutes'] + 1) * 90

    # Normalize fixture difficulty
    teams['fixture_difficulty'] = (teams['strength_attack_home'] + teams['strength_attack_away']) / 2

    # Merge with proper column naming
    players = players.merge(teams[['id', 'fixture_difficulty']], 
                            left_on='team', 
                            right_on='id', 
                            how='left')

    # Expected minutes calculation
    players['expected_minutes'] = players.apply(
        lambda row: row['minutes'] if row['news'] == '' else row['minutes'] / 2, 
        axis=1
    )
    return players

# Add opposition team dynamically
def assign_opposition(players, fixtures_df):
    team_id_to_name = teams.set_index('id')['name'].to_dict()
    fixtures_df['team_h_name'] = fixtures_df['team_h'].map(team_id_to_name)
    fixtures_df['team_a_name'] = fixtures_df['team_a'].map(team_id_to_name)

    # Map players to their opposition team based on fixtures
    def get_opposition_team(player_row):
        player_team = player_row['team']
        for _, fixture in fixtures_df.iterrows():
            if fixture['team_h'] == player_team:
                return fixture['team_a']  # Away team is the opposition
            elif fixture['team_a'] == player_team:
                return fixture['team_h']  # Home team is the opposition
        return None

    players['opposition_team'] = players.apply(get_opposition_team, axis=1)
    return players

# Helper Functions for Criteria Calculations
def calculate_recent_form(player):
    """Calculate the average points of the player in the last few games."""
    # Note: This would require additional API calls or historical data
    # Fallback to current form if no detailed data available
    return float(player['form']) * 2  # Multiply by 2 to give more weight

def calculate_fixture_difficulty(player, teams):
    """Calculate fixture difficulty based on team strength."""
    # Use team's overall strength as a proxy
    team_strength = teams.loc[teams['id'] == player['team'], 'strength'].values[0]
    return -team_strength  # Negative because lower strength means harder fixture

def calculate_absences_opposition(player, players_df, key_players_by_team):
    """
    Estimate impact of opposition team's injuries based on key players.

    Args:
    player: Row from the players DataFrame representing a single player.
    players_df: DataFrame containing player data, including injury status.
    key_players_by_team: Dictionary of teams (by ID) with their key players.

    Returns:
    float: Points adjustment based on opposition team's key player injuries.
    """
    # Get the opposition team's ID from the player data
    opposition_team_id = player.get('opposition_team')
    if not opposition_team_id or opposition_team_id not in key_players_by_team:
        return 0  # Return 0 if no valid opposition team ID

    # Get the list of key players for the opposition team
    key_players = key_players_by_team[opposition_team_id]

    # Find injured key players by matching names and checking injury status
    injured_key_players = []
    for key_player in key_players:
        matched_players = players_df[
            (players_df['web_name'].str.lower() == key_player.lower()) &
            (players_df['news'] != '')  # Non-empty news indicates an injury
        ]
        if not matched_players.empty:
            injured_key_players.append(key_player)

    # Scoring logic based on the number of injured key players
    num_injured = len(injured_key_players)
    if num_injured == 1:
        return 1  # Small bonus for one injured key player
    elif num_injured == 2:
        return 2  # Larger bonus for two injured key players
    elif num_injured >= 3:
        return 3  # Maximum bonus for three or more injured key players

    return 0  # No injured key players

def calculate_time_in_round(player, fixtures_df):
    """
    Adjust points based on game timing.
    - Attackers' and midfielders' points are halved if their match is the first in the round.
    - Goalkeepers' and defenders' points are doubled if their match is the first in the round.

    Args:
    player: Row from the players DataFrame representing a single player.
    fixtures_df: DataFrame containing fixture information for the current gameweek.

    Returns:
    float: Adjustment to the player's points.
    """
    # Sort fixtures by kickoff time and identify the first match
    first_match = fixtures_df.sort_values(by='kickoff_time').iloc[0]
    first_match_teams = {first_match['team_h'], first_match['team_a']}
    
    # Check if the player's team is in the first match
    if player['team'] in first_match_teams:
        # Halve points for attackers (4) and midfielders (3)
        if player['element_type'] in [3, 4]:  # 3 for midfielder, 4 for attacker
            return -player['projected_points'] / 2  # Negative adjustment to halve points
        # Double points for goalkeepers (1) and defenders (2)
        elif player['element_type'] in [1, 2]:  # 1 for goalkeeper, 2 for defender
            return player['projected_points']  # Positive adjustment to double points

# Additional Criteria Suggestions
def calculate_home_advantage(player):
    """Players perform better when playing at home."""
    return 0

def calculate_team_momentum(player):
    """Higher recent team performance boosts individual player performance."""
    return 0

# Objective Functions
def calculate_gk_points(player, players, key_players_by_team, fixtures_df):
    """Calculate projected points for a Goalkeeper."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player, players, key_players_by_team),
        calculate_time_in_round(player, fixtures_df),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_def_points(player, players, key_players_by_team, fixtures_df):
    """Calculate projected points for a Defender."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player, players, key_players_by_team),
        calculate_time_in_round(player, fixtures_df),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_mid_points(player, players, key_players_by_team, fixtures_df):
    """Calculate projected points for a Midfielder."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player, players, key_players_by_team),
        calculate_time_in_round(player, fixtures_df),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def calculate_fwd_points(player, players, key_players_by_team, fixtures_df):
    """Calculate projected points for a Forward."""
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player, teams),
        calculate_absences_opposition(player, players, key_players_by_team),
        calculate_time_in_round(player, fixtures_df),
        calculate_home_advantage(player),
        calculate_team_momentum(player)
    ])

def predict_points(players, key_players_by_team):
    """Calculate projected points for all players based on position."""
    for index, player in players.iterrows():
        if player['element_type'] == 1:  # Goalkeeper
            players.at[index, 'projected_points'] = calculate_gk_points(player, players, key_players_by_team, fixtures_df)
        elif player['element_type'] == 2:  # Defender
            players.at[index, 'projected_points'] = calculate_def_points(player, players, key_players_by_team, fixtures_df)
        elif player['element_type'] == 3:  # Midfielder
            players.at[index, 'projected_points'] = calculate_mid_points(player, players, key_players_by_team, fixtures_df)
        elif player['element_type'] == 4:  # Forward
            players.at[index, 'projected_points'] = calculate_fwd_points(player, players, key_players_by_team, fixtures_df)


if __name__ == "__main__":
    current_gameweek = get_current_gameweek()
    players = preprocess_data(players, teams)
    players = assign_opposition(players, get_current_fixtures(current_gameweek))
    players['projected_points'] = 0.0  # Initialize projected points column

    predict_points(players, key_players_by_team)

    top_players = players.sort_values(by='projected_points', ascending=False)[
        ['first_name', 'second_name', 'team', 'element_type', 'projected_points', 'web_name']
    ].head(50)

    fig = px.bar(
        top_players, 
        x='web_name', 
        y='projected_points', 
        color='element_type', 
        title='Top 50 FPL Players - Projected Points'
    )
    fig.show()

