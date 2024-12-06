import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch FPL data from API
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player and team data
players = pd.DataFrame(data['elements'])  # Player data
teams = pd.DataFrame(data['teams'])  # Team data

players = players.merge(teams[['id', 'name']], left_on='team', right_on='id', how='left')

# Calculate average cost and points per game for each team
players['cost'] = players['now_cost'] / 10  # Convert cost to millions
players['points_per_game'] = pd.to_numeric(players['points_per_game'], errors='coerce')  # Ensure numeric
team_summary = players.groupby('name').agg(
    avg_cost=('cost', 'mean'),
    avg_points=('points_per_game', 'mean')
).reset_index()

# Create the scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(
    data=team_summary, 
    x='avg_cost', 
    y='avg_points', 
    scatter_kws={'color': 'blue', 's': 50}, 
    line_kws={'color': 'red'}
)

for i in range(len(team_summary)):
    plt.text(
        x=team_summary['avg_cost'][i] + 0.02,  # Offset for clarity
        y=team_summary['avg_points'][i],
        s=team_summary['name'][i],
        fontsize=9
    )

plt.title('Average Points per Game vs Average Cost Across Teams', fontsize=14)
plt.xlabel('Average Cost (millions in dollars $)', fontsize=12)
plt.ylabel('Average Points per Game', fontsize=12)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
