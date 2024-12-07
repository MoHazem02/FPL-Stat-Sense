import asyncio
from fpl import FPL
import aiohttp

async def get_team_name(fpl, team_id):
    """
    Fetches the name of the team based on the team_id.
    """
    team = await fpl.get_team(team_id, return_json=True)
    return team['name']

async def get_matches_for_gameweek(gameweek):
    """
    Fetches and displays the matches for the specified gameweek using the FPL library.
    """
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        fixtures = await fpl.get_fixtures_by_gameweek(gameweek, return_json=True)

        print(f"Matches for Gameweek {gameweek}:")
        for fixture in fixtures:
            home_team_name = await get_team_name(fpl, fixture['team_h'])
            away_team_name = await get_team_name(fpl, fixture['team_a'])
            print(f"{home_team_name} vs {away_team_name} at {fixture['kickoff_time']}")


# Example usage for gameweek 15
if __name__ == "__main__":
    gameweek = 15
    asyncio.run(get_matches_for_gameweek(gameweek))