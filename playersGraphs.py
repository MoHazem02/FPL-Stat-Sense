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
players_filtered = players[players['ppm'] > 0]
top_20_players = players_filtered.nlargest(20, 'ppm')


plt.figure(figsize=(10, 6))
sns.barplot(x='ppm', y='web_name', data=top_20_players, palette='viridis')

plt.title('Top 20 FPL Players Based on Points Per Million (PPM)', fontsize=16)
plt.xlabel('Points Per Million (PPM)', fontsize=12)
plt.ylabel('Player Name', fontsize=12)
plt.tight_layout()
plt.show()
