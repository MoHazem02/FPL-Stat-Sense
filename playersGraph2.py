import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch data from FPL API
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player and team data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

# Calculate Points Per Game (PPG)
players['ppg'] = players.apply(lambda row: row['total_points'] / row['minutes'] * 90 if row['minutes'] > 0 else 0, axis=1)

# Filter players who have played at least 5 matches (450 minutes) and have PPG > 0
players_filtered = players[(players['minutes'] >= 450) & (players['ppg'] > 0)]  # 450 minutes = 5 matches worth of minutes

# Separate players by position
gk_and_defenders = players_filtered[players_filtered['element_type'].isin([1, 2])]  # Goalkeepers (1) and Defenders (2)
midfielders_and_forwards = players_filtered[players_filtered['element_type'].isin([3, 4])]  # Midfielders (3) and Forwards (4)

# Get top 50 Goalkeepers and Defenders, top 50 Midfielders and Forwards by PPG
top_50_gk_and_defenders = gk_and_defenders.nlargest(50, 'ppg')
top_50_midfielders_and_forwards = midfielders_and_forwards.nlargest(50, 'ppg')

# Get the weakest attacking teams (fewest goals scored) for home matches
weakest_attack_teams_home = teams.sort_values('strength_attack_home').head(5)  # Weakest attacking teams at home
weakest_attack_team_ids_home = weakest_attack_teams_home['id']

# Get the weakest attacking teams (fewest goals scored) for away matches
weakest_attack_teams_away = teams.sort_values('strength_attack_away').head(5)  # Weakest attacking teams away
weakest_attack_team_ids_away = weakest_attack_teams_away['id']

# Get the weakest defensive teams (most goals conceded) for away matches
weakest_defense_teams_away = teams.sort_values('strength_defence_away', ascending=False).head(5)  # Weakest defensive teams away
weakest_defense_team_ids_away = weakest_defense_teams_away['id']

# Get the weakest defensive teams (most goals conceded) for home matches
weakest_defense_teams_home = teams.sort_values('strength_defence_home', ascending=False).head(5)  # Weakest defensive teams at home
weakest_defense_team_ids_home = weakest_defense_teams_home['id']

# Filter Goalkeepers and Defenders playing away against weak attacking teams at home or home against weak attacking teams away
gk_defender_playing_away_against_weak_attack_home = top_50_gk_and_defenders[
    (top_50_gk_and_defenders['team'].isin(weakest_attack_team_ids_home))  # Away against weak attack home teams
]

gk_defender_playing_home_against_weak_attack_away = top_50_gk_and_defenders[
    (top_50_gk_and_defenders['team'].isin(weakest_attack_team_ids_away))  # Home against weak attack away teams
]

# Combine both conditions for Goalkeepers and Defenders
gk_defender_filtered = pd.concat([gk_defender_playing_away_against_weak_attack_home, 
                                  gk_defender_playing_home_against_weak_attack_away])

# Sort by PPG in descending order
gk_defender_filtered = gk_defender_filtered.sort_values('ppg', ascending=False)

# Filter Midfielders and Forwards playing home against weak defensive teams away or away against weak defensive teams home
midfielder_forward_playing_home_against_weak_defense_away = top_50_midfielders_and_forwards[
    (top_50_midfielders_and_forwards['team'].isin(weakest_defense_team_ids_away))  # Home against weak defense away teams
]

midfielder_forward_playing_away_against_weak_defense_home = top_50_midfielders_and_forwards[
    (top_50_midfielders_and_forwards['team'].isin(weakest_defense_team_ids_home))  # Away against weak defense home teams
]

# Combine both conditions for Midfielders and Forwards
midfielder_forward_filtered = pd.concat([midfielder_forward_playing_home_against_weak_defense_away, 
                                         midfielder_forward_playing_away_against_weak_defense_home])

# Sort by PPG in descending order
midfielder_forward_filtered = midfielder_forward_filtered.sort_values('ppg', ascending=False)

# Now let's plot the graph
plt.figure(figsize=(14, 8))

# Plot: PPG of Goalkeepers and Defenders playing against weak attack teams (home vs away)
plt.subplot(2, 2, 1)
sns.barplot(x='ppg', y='web_name', data=gk_defender_filtered, palette='Blues')
plt.title('Goalkeepers and Defenders (Home vs Away) Against Weak Attack Teams')
plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Player Name')

# Plot: PPG of Midfielders and Forwards playing against weak defense teams (away vs home)
plt.subplot(2, 2, 2)
sns.barplot(x='ppg', y='web_name', data=midfielder_forward_filtered, palette='Oranges')
plt.title('Midfielders and Forwards (Away vs Home) Against Weak Defense Teams')
plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Player Name')

# Add gridlines and adjust layout
plt.tight_layout()
plt.show()
