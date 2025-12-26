import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb

def convert_to_heatmap(df, x_col, y_col, endX_col, endY_col, bins=(60, 40), cmap='Reds'):
    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')
    
    pitch = Pitch(pitch_type='statsbomb', 
                  pitch_color = 'grass', line_color='#c7d5cc')

    x = df[x_col]
    y = df[y_col]
    
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()
    
    # Heatmap based on all touch locations
    heatmap = pitch.bin_statistic(x, y, statistic='count', bins=bins)
    pitch.heatmap(heatmap, ax=ax, cmap=cmap, alpha=0.5)
    
    # Use a for loop to plot each event
    for idx in df.index:
        # Only draw lines for events that have an end location (Passes and Carries)
        if pd.notnull(df.loc[idx, endX_col]):
            color = 'green' if df.loc[idx, 'outcome'] == 'Successful' else 'red'
            plt.plot((df.loc[idx, x_col], df.loc[idx, endX_col]), 
                     (df.loc[idx, y_col], df.loc[idx, endY_col]), 
                     color=color, linewidth=1.5, alpha=0.6)
        
        # Plot a scatter point for every touch
        #plt.scatter(df.loc[idx, x_col], df.loc[idx, y_col], color='white', s=10, alpha=0.5)

    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.title('Florian Wirtz Touch & Pass Map', color='white', size=20)
    plt.show()

def load_player_events(player_name, season_id):
    matches = sb.matches(competition_id=9, season_id=season_id)
    team_name = "Bayer Leverkusen"
    player_matches = matches[(matches['home_team'] == team_name) | (matches['away_team'] == team_name)]
    match_id = player_matches.iloc[0]['match_id']
    events = sb.events(match_id=match_id)

    # Define Touch types
    touch_types = ['Pass', 'Carry', 'Ball Receipt*', 'Dribble', 'Shot', 'Ball Recovery']
    
    player_events = events[(events['player'] == player_name) & (events['type'].isin(touch_types))].copy()

    player_events = player_events.dropna(subset=['location'])
    player_events['x'] = player_events['location'].str[0]
    player_events['y'] = player_events['location'].str[1]


    player_events['endX'] = player_events['pass_end_location'].str[0].fillna(player_events['carry_end_location'].str[0])
    player_events['endY'] = player_events['pass_end_location'].str[1].fillna(player_events['carry_end_location'].str[1])
    
    # Normalize outcomes for coloring
    player_events['outcome'] = 'Successful'
    if 'pass_outcome' in player_events.columns:
        player_events.loc[player_events['pass_outcome'].notnull(), 'outcome'] = 'Unsuccessful'
    if 'carry_outcome' in player_events.columns:
        player_events.loc[player_events['carry_outcome'].notnull(), 'outcome'] = 'Unsuccessful'

    convert_to_heatmap(player_events, 'x', 'y', 'endX', 'endY')

if __name__ == "__main__":
    load_player_events(player_name="Florian Wirtz", season_id=281)