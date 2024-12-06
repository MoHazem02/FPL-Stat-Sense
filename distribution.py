import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define FPL API URL
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

try:
    response = requests.get(FPL_API_URL)
    response.raise_for_status()  # Raise an error for bad HTTP responses
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
    exit()

# Extract relevant data
players = pd.DataFrame(data['elements'])  # Player data
positions = pd.DataFrame(data['element_types'])  # Position mapping

# Merge player data with positions
players = players.merge(positions[['id', 'singular_name_short']], left_on='element_type', right_on='id', how='left')
players.rename(columns={'singular_name_short': 'position'}, inplace=True)


players['games_played'] = players['minutes'] / 90  # Estimate games played
players['games_played'].fillna(0, inplace=True)  # Replace NaN with 0
players['total_points'].fillna(0, inplace=True)  # Replace NaN with 0

players_filtered = players[
    (players['games_played'] >= 2) &  # Minimum 2 games played
    (players['total_points'] > 0) &  # Minimum total points
    (players['minutes'] > 0)         # Avoid division by zero
]

# Calculate Points Per 90 Minutes (PP90)
players_filtered['pp90'] = players_filtered.apply(
    lambda x: (x['total_points'] / x['minutes']) * 90 if x['minutes'] > 0 else 0, axis=1
)

# Select the top 50 players based on PP90
top_50_players = players_filtered.nlargest(50, 'pp90')

# Plot 1: Distribution of Positions Among Top 50 Players
plt.figure(figsize=(12, 6))
position_counts = top_50_players['position'].value_counts()  # Get position counts
sns.barplot(x=position_counts.index, y=position_counts.values, palette='viridis')

plt.title('Distribution of Positions Among Top 50 Players (Based on Points Per match)', fontsize=16)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Number of Players', fontsize=12)
plt.bar_label(plt.gca().containers[0])  # Add value labels to bars
plt.grid(axis='y', alpha=0.3)

# Plot 2: PP90 Distribution by Position for Top 50 Players
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_50_players, x='position', y='pp90', palette='viridis')

plt.title(' Distribution by Position for Top 50 Players (Based on Points Per match)', fontsize=16)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Points Per 90 Minutes', fontsize=12)
plt.grid(axis='y', alpha=0.3)


plt.tight_layout() 
plt.show()
