import asyncio
from fpl import FPL
import aiohttp
import plotly.graph_objects as go
from fixture_difficulty import get_team_ranking_adjustment, get_team_xG

async def get_current_gameweek(fpl):
    gameweeks = await fpl.get_gameweeks()
    for gw in gameweeks:
        if not gw.finished:
            return gw.id
    return None

async def get_team_name(fpl, team_id):
    team = await fpl.get_team(team_id, return_json=True)
    return team['name']

async def get_league_standings(fpl):
    standings = await fpl.get_standings()
    league_table = {}
    for index, team in enumerate(standings['standings']['results'], start=1):
        league_table[team['entry']] = index  # Team ID mapped to its rank
    return league_table

async def get_player_data(fpl, gameweek):
    players = await fpl.get_players(return_json=True)
    fixtures = await fpl.get_fixtures_by_gameweek(gameweek, return_json=True)

    player_data = []
    for player in players:
        for fixture in fixtures:
            if fixture['team_h'] == player['team']:
                opposition_team = fixture['team_a']
                is_home = True
                break
            elif fixture['team_a'] == player['team']:
                opposition_team = fixture['team_h']
                is_home = False
                break
        else:
            continue

        opposition_team_name = await get_team_name(fpl, opposition_team)
        

        player_data.append({
            "name": player['web_name'],
            "opposition_team": opposition_team_name,
            "is_home": is_home,
            "recent_form": player['form'],
            "position": player['element_type'],
            "team": await get_team_name(fpl, player['team'])
        })
    
    return player_data

def calculate_recent_form(player):
    return float(player['recent_form']) * 2

def calculate_fixture_difficulty(player):
    opposition_team = player["opposition_team"]
    return get_team_ranking_adjustment(opposition_team)

def calculate_home_advantage(player):
    return 2 if player["is_home"] else 0

def calculate_team_xG(player):
    return get_team_xG(player["team"])

def predict_fwd_points(player):
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player),
        calculate_home_advantage(player),
        calculate_team_xG(player)
    ])

def predict_mid_points(player):
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player),
        calculate_home_advantage(player),
        calculate_team_xG(player)
    ])

def predict_def_points(player):
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player),
        calculate_home_advantage(player)
    ])

def predict_gk_points(player):
    return sum([
        calculate_recent_form(player),
        calculate_fixture_difficulty(player),
        calculate_home_advantage(player)
    ])

def calculate_projected_points(player):
    position = player["position"]
    if position == 4:  # Forward
        return predict_fwd_points(player)
    elif position == 3:  # Midfielder
        return predict_mid_points(player)
    elif position == 2:  # Defender
        return predict_def_points(player)
    elif position == 1:  # Goalkeeper
        return predict_gk_points(player)
    return 0.0

def display_top_players(player_data):
    sorted_players = sorted(player_data, key=lambda x: x['projected_points'], reverse=True)

    # Select the top 50 players
    top_players = sorted_players[:50]

    # Prepare data for Plotly
    names = [player['name'] for player in top_players]
    points = [player['projected_points'] for player in top_players]

    # Plot the data
    fig = go.Figure(data=[
        go.Bar(x=names, y=points, text=points, textposition='auto')
    ])
    fig.update_layout(
        title="Top 10 Players by Projected Points",
        xaxis_title="Players",
        yaxis_title="Projected Points",
        template="plotly_dark"
    )
    fig.show()

async def main():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)

        current_gameweek = await get_current_gameweek(fpl)
        if current_gameweek:
            print(f"Current Gameweek: {current_gameweek}")
            player_data = await get_player_data(fpl, current_gameweek)

            # Add projected points for each player
            for player in player_data:
                player['projected_points'] = calculate_projected_points(player)

            # Display top players using Plotly
            display_top_players(player_data)
        else:
            print("Unable to determine the current gameweek.")

asyncio.run(main())
