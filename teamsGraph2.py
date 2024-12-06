import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch FPL data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player and team data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

# Add team names to players
players['points_per_game'] = pd.to_numeric(players['points_per_game'], errors='coerce')
team_summary = players.groupby('team').agg(
    team_ppg=('points_per_game', 'mean'),  # Points per game by team
    fpl_ppg=('total_points', 'mean')      # Average FPL points per game
).reset_index()


team_summary['name'] = team_summary['team'].map(teams.set_index('id')['name'])


plt.figure(figsize=(10, 8))

plt.scatter(team_summary['fpl_ppg'], team_summary['team_ppg'], color='blue', label='Teams')

for _, row in team_summary.iterrows():
    
    plt.text(row['fpl_ppg'] + 0.1, row['team_ppg'], row['name'], fontsize=9, ha='left', va='center')

plt.plot([team_summary['fpl_ppg'].min(), team_summary['fpl_ppg'].max()],
         [team_summary['team_ppg'].min(), team_summary['team_ppg'].max()],
         linestyle='--', color='gray', label='Trend Line')

plt.axhline(y=team_summary['team_ppg'].mean(), color='black', linestyle='-', alpha=0.5, label='Avg PPG')
plt.axvline(x=team_summary['fpl_ppg'].mean(), color='black', linestyle='-', alpha=0.5, label='Avg FPL PPG')

plt.title('Points Per Game (PPG) vs FPL Points Per Game (FPL PPG)', fontsize=16)
plt.xlabel('FPL Points Per Game (FPL PPG)', fontsize=12)
plt.ylabel('Points Per Game (PPG)', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
