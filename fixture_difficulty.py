import requests
from bs4 import BeautifulSoup
import pandas as pd

# Global variable to store standings data
standings_data = None

def get_premier_league_standings():
    global standings_data
    
    if standings_data is not None:
        return standings_data  # Return the cached data if available
    
    # URL for Premier League standings
    url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
    
    # Send a request to the webpage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the standings table using the specific ID
        standings_table = soup.find('table', {'id': 'results2024-202591_overall'})
        
        if not standings_table:
            print("Could not find the standings table")
            return None
        
        # Extract specific columns
        rows = []
        body_rows = standings_table.find('tbody').find_all('tr')
        for tr in body_rows:
            # Extract rank, team, and xG
            rank = tr.find('th').get_text(strip=True)
            team = tr.find_all('td')[0].get_text(strip=True)
            xg = tr.find_all('td')[11].get_text(strip=True)
            
            if team == 'Ipswich Town':
                team = 'Ipswich'
            elif team == "Nott'ham Forest":
                team = "Nott'm Forest"
            elif team == 'Tottenham':
                team = 'Spurs'
            elif team == 'Manchester Utd':
                team = 'Man Utd'
            elif team == 'Manchester City':
                team = 'Man City'
            elif team == 'Newcastle Utd':
                team = 'Newcastle'

            rows.append([rank, team, xg])
        
        # Create a DataFrame
        standings_data = pd.DataFrame(rows, columns=['Rank', 'Team', 'xG'])
        
        return standings_data
    
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None

def get_team_ranking_adjustment(team_name):
    """
    Returns a point adjustment based on the team's ranking in the Premier League.
    
    Args:
        team_name (str): Name of the team to check
    
    Returns:
        int: Point adjustment 
        -4 if team is in top 5
        -3 if team is in top 7
        -1 if team is in top 15
        0 otherwise
    """
    # Get the standings
    standings = get_premier_league_standings()
    
    while(standings is None):
        standings = get_premier_league_standings()
    team_row = standings[standings['Team'] == team_name]
    
    # Convert rank to integer
    rank = int(team_row['Rank'].values[0])
    
    # Apply point adjustments based on ranking
    if rank <= 5:
        return -4
    elif rank <= 7:
        return -3
    elif rank <= 15:
        return -1
    else:
        return 0

def get_team_xG(team_name):
    """
    Returns the expected goals (xG) for a given team in the Premier League.
    
    Args:
        team_name (str): Name of the team to check
    
    Returns:
        float: Expected goals (xG) value for the team
    """
    # Get the standings
    standings = get_premier_league_standings()
    
    # Find the team's xG value
    team_row = standings[standings['Team'] == team_name]
    
    # Convert xG to float
    xg = float(team_row['xG'].values[0])
    
    return xg

if __name__ == "__main__":
    # Display the standings
    standings = get_premier_league_standings()
    if standings is not None:
        # Pretty print the standings
        print(standings)
        
        # Example of using the new ranking adjustment function
        print("\nTeam Ranking Adjustments:")
        example_teams = ["Bournemouth", "Liverpool", "Arsenal", "Tottenham", "Crystal Palace"]
        for team in example_teams:
            adjustment = get_team_ranking_adjustment(team)
            print(f"{team}: {adjustment} point adjustment")
