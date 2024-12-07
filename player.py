import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch FPL data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player data
players = pd.DataFrame(data['elements'])

# Calculate Points Per Million (PPM)
players['ppm'] = players['total_points'] / players['now_cost']  # now_cost is in tenths of a million
print(players['expected_goal_involvements'])
# Filter players who have valid xGI and total points > 0
players['expected_goal_involvements'] = pd.to_numeric(players['expected_goal_involvements'], errors='coerce')

# Drop rows with invalid `expected_goal_involvements`
players = players[~players['expected_goal_involvements'].isna()]

# Filter players with valid expected goal involvements and total points
players_filtered = players[
    (players['expected_goal_involvements'] > 0) & 
    (players['total_points'] > 0)
]



# Scatter Plot: xGI vs Total Points
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='expected_goal_involvements',
    y='total_points',
    hue='element_type',  # Highlight by position
    data=players_filtered,
    palette='Set2',
    s=100,  # Marker size
    alpha=0.7
)

# Add labels and title
plt.title('Expected Goal Involvement vs Total Points', fontsize=16)
plt.xlabel('Expected Goal Involvement (xGI)', fontsize=12)
plt.ylabel('Total Points', fontsize=12)
plt.legend(title='Position', loc='upper left', labels=['GKP', 'DEF', 'MID', 'FWD'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Display the plot
plt.show()
