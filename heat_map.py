import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb

def convert_to_heatmap(df, x_col, y_col, bins=(10, 10), cmap='Reds'):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    
    x = df[x_col] * 1.2
    y = df[y_col] * 0.8
    
    heatmap = pitch.bin_statistic(x, y, statistic='count', bins=bins)
    pitch.heatmap(heatmap, ax=ax, cmap=cmap, alpha=0.7)
    
    plt.title('Heatmap of ' + x_col + ' vs ' + y_col)
    plt.show()

if __name__ == "__main__":
    #convert_to_heatmap()
    competitions = sb.competitions()
    matches = sb.matches(competition_id=9, season_id=281)
    team_name = "Bayer Leverkusen"
    player_matches = matches[(matches['home_team'] == team_name) | (matches['away_team'] == team_name)]
    match_id = player_matches.iloc[0]['match_id']
    events = sb.events(match_id=match_id)
    player_name = "Florian Wirtz"  # Or another player
    player_events = events[events['player'] == player_name]

    print(player_events['type'].value_counts())
