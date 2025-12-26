import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb

def get_matches(competition_id, season_id, team_name):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    team_matches = matches[(matches['home_team'] == team_name) | (matches['away_team'] == team_name)]
    return team_matches

if __name__ == "__main__":
    competition_id = 9  # Bundesliga
    season_id = 281      # 2023/2024 Season
    team_name = "Bayer Leverkusen"
    
    matches = get_matches(competition_id, season_id, team_name)
    print(matches[['home_team', 'away_team', 'match_id']])