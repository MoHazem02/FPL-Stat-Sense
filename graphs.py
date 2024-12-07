import pandas as pd
import numpy as np
import seaborn as sns
import requests

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(FPL_API_URL)
data = response.json()

# Extract player data
players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

# Add PPG column
players['ppg'] = players.apply(
    lambda row: row['total_points'] / row['minutes'] * 90 if row['minutes'] > 0 else 0,
    axis=1
)

# Filter relevant players (played > 450 minutes, PPG > 0)
players_filtered = players[(players['minutes'] >= 360) & (players['ppg'] > 0)]

# Extract positions
positions = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
players_filtered['position'] = players_filtered['element_type'].map(positions)

# --- 1. Descriptive Statistics ---
descriptive_stats = players_filtered[['ppg', 'total_points', 'minutes', 'now_cost']].describe()

# --- 2. Visualization ---
plt.figure(figsize=(14, 10))

# Histogram for PPG
plt.subplot(2, 2, 1)
sns.histplot(players_filtered['ppg'], kde=True, bins=20, color='blue')
plt.title('Distribution of Points Per Game (PPG)')
plt.xlabel('PPG')
plt.ylabel('Frequency')

# Box Plot by Position
plt.subplot(2, 2, 2)
sns.boxplot(x='position', y='ppg', data=players_filtered, palette='Set2')
plt.title('PPG by Position')
plt.xlabel('Position')
plt.ylabel('PPG')

# Scatter Plot: Minutes Played vs. PPG
plt.subplot(2, 2, 3)
sns.scatterplot(x='minutes', y='ppg', hue='position', data=players_filtered, palette='Dark2')
plt.title('Minutes Played vs. PPG')
plt.xlabel('Minutes Played')
plt.ylabel('PPG')

# Scatter Plot: Cost vs. Total Points
plt.subplot(2, 2, 4)
sns.scatterplot(x='now_cost', y='total_points', hue='position', data=players_filtered, palette='Set1')
plt.title('Cost vs. Total Points')
plt.xlabel('Cost (in millions)')
plt.ylabel('Total Points')

plt.tight_layout()
plt.show()

# --- 3. Correlation Analysis ---
correlation_matrix = players_filtered[['ppg', 'total_points', 'minutes', 'now_cost']].corr()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# --- 4. Group-wise Analysis ---
position_stats = players_filtered.groupby('position').agg({
    
    
    'minutes': ['mean', 'median', 'var', 'std', 'max']
})



# Display Descriptive Statistics
print("Descriptive Statistics:")
print(descriptive_stats)

print("\nGroup-wise Averages by Position:")
print(position_stats)

# --- 5. Top Performers ---
top_performers = players_filtered.nlargest(10, 'ppg')[['web_name', 'ppg', 'total_points', 'minutes', 'position', 'now_cost']]
print("\nTop Performers by PPG:")
print(top_performers)
correlation_ppg_total_points = correlation_matrix.loc['ppg', 'total_points']
correlation_cost_total_points = correlation_matrix.loc['now_cost', 'total_points']
correlation_minutes_cost = correlation_matrix.loc['minutes', 'now_cost']

# Print results
print(f"Correlation between Total Points and PPG: {correlation_ppg_total_points:.2f}")
print(f"Correlation between Cost and Total Points: {correlation_cost_total_points:.2f}")
print(f"Correlation between Minutes and Cost: {correlation_minutes_cost:.2f}")
