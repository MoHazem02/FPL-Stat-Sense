import pandas as pd
import numpy as np
import seaborn as sns
import requests
import matplotlib.pyplot as plt

# Load data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player data
players = pd.DataFrame(data['elements'])
positions = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}

# Add columns for PPG and Position
players['ppg'] = players.apply(
    lambda row: row['total_points'] / row['minutes'] * 90 if row['minutes'] > 0 else 0,
    axis=1
)
players['position'] = players['element_type'].map(positions)

# Add Goal Involvements (Goals + Assists)
players['goal_involvements'] = players['goals_scored'] + players['assists']

# Filter relevant players (played > 450 minutes, PPG > 0)
players_filtered = players[(players['minutes'] >= 360) & (players['ppg'] > 0)]

# --- Descriptive Statistics ---
descriptive_stats = players_filtered[['ppg', 'total_points', 'goal_involvements', 'minutes', 'now_cost']].describe()

# --- Visualization ---
plt.figure(figsize=(14, 12))

# Scatter Plot: PPG vs. Expected Goal Involvements
plt.subplot(2, 1, 1)
sns.scatterplot(x='expected_goal_involvements', y='ppg', hue='position', data=players_filtered, palette='Set1')
plt.title('PPG vs. Expected Goal Involvements')
plt.xlabel('Expected Goal Involvements')
plt.ylabel('PPG')

# Scatter Plot: PPG vs. Goal Involvements
plt.subplot(2, 1, 2)
sns.scatterplot(x='goal_involvements', y='ppg', hue='position', data=players_filtered, palette='Set2')
plt.title('PPG vs. Goal Involvements')
plt.xlabel('Goal Involvements')
plt.ylabel('PPG')

plt.tight_layout()
plt.show()

# --- Correlation Analysis ---
correlation_matrix = players_filtered[['ppg', 'expected_goal_involvements', 'goal_involvements', 'minutes', 'now_cost']].corr()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# --- Group-wise Analysis ---
position_stats = players_filtered.groupby('position').agg({
    'ppg': ['mean', 'median', 'std', 'max'],
    'expected_goal_involvements': ['mean', 'median', 'std', 'max'],
    'goal_involvements': ['mean', 'median', 'std', 'max']
})

# Display Descriptive Statistics
print("Descriptive Statistics:")
print(descriptive_stats)

print("\nGroup-wise Analysis by Position:")
print(position_stats)

# --- Top Performers ---
top_performers = players_filtered.nlargest(10, 'ppg')[['web_name', 'ppg', 'expected_goal_involvements', 'goal_involvements', 'minutes', 'position', 'now_cost']]
print("\nTop Performers by PPG:")
print(top_performers)
