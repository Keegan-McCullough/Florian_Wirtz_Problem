import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


def convert_to_heatmap(df, x_col, y_col, endX_col, endY_col, bins=(60, 40), cmap='Reds', game_stats=None):
    """Plot KDE heatmap with pass arrows and optional game stats side panel."""
    fig, (ax, ax_stats) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')
    ax_stats.patch.set_facecolor('#22312b')
    ax_stats.axis('off')
    
    pitch = Pitch(pitch_type='statsbomb', 
                  pitch_color = 'grass', line_color='#c7d5cc')

    x = df[x_col]
    y = df[y_col]
    
    pitch.draw(ax=ax)
    
    # KDE plot heatmap based on all touch locations
    sns.kdeplot(x=x, y=y, ax=ax, cmap=plt.cm.inferno, fill=True, alpha=0.5, levels=6, thresh=0.1)
    # Use a for loop to plot each event
    for idx in df.index:
        # Only draw arrows for events that have an end location for Passes
        if pd.notnull(df.loc[idx, endX_col]):
            color = 'green' if df.loc[idx, 'outcome'] == 'Successful' else 'red'
            ax.annotate('', xy=(df.loc[idx, endX_col], df.loc[idx, endY_col]),
                       xytext=(df.loc[idx, x_col], df.loc[idx, y_col]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
            
    for idx in df.index:
        if df.loc[idx, 'pass_goal_assist'] == 'True':
            plot_icon(ax, 'icons/cleats.png',
                      df.loc[idx, x_col], df.loc[idx, y_col], zoom=0.04)
        elif df.loc[idx, 'type'] == 'Shot':
            if df.loc[idx, 'shot_outcome'] == 'Goal':
                plot_icon(ax, 'icons/soccer_ball.png',
                      df.loc[idx, x_col], df.loc[idx, y_col], zoom=0.03)
            else:
                plot_icon(ax, 'icons/close.png',
                      df.loc[idx, x_col], df.loc[idx, y_col], zoom=0.025)
                
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 90)
    ax.invert_yaxis()
    ax.set_title('Florian Wirtz Touch & Pass Map', color='white', size=20)

    # Side panel: game stats if provided
    if game_stats is not None and not game_stats.empty:
        y0 = 0.9
        dy = 0.12
        ax_stats.text(0.05, y0, 'Game Stats', color='white', fontsize=14, weight='bold')
        stats_order = ['rating', 'min', 'G', 'A', 'shots', 'drib', 'touches']
        labels = {
            'rating': 'Rating',
            'min': 'Minutes Played',
            'G': 'Goals',
            'A': 'Assists',
            'shots': 'Shots',
            'drib': 'Dribbles',
            'touches': 'Touches'
        }
        for i, key in enumerate(stats_order):
            val = game_stats.get(key, '')
            if pd.isna(val):
                val = 0
            if key == 'rating':
                val = str(val) + '/10'
            else:
                val = int(val)
            ax_stats.text(0.05, y0 - (i + 1) * dy, f"{labels[key]}: {val}", color='white', fontsize=12)
    else:
        ax_stats.text(0.05, 0.5, 'Game stats not found.', color='white', fontsize=12)

    plt.tight_layout()
    plt.show()

def load_game_stats(game_label, csv_path='game_stats.csv'):
    """Load a single row of game stats by label (e.g., 'Augsburg (A)')."""
    try:
        stats_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return pd.Series()
    game_label = process_names(game_label)
    row = stats_df[stats_df['game'] == game_label]
    return row.squeeze() if not row.empty else pd.Series()

def process_names(name):
    if 'ö' in name:
        name = name.replace('ö', 'o')
    return name

def plot_icon(ax,img_path, x, y, zoom=0.05):
    img = mpimg.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)


def load_player_events(player_name, match_id, team_name="Bayer Leverkusen"):
    # Try to resolve opponent and home/away from matches table for Bundesliga season 281
    try:
        matches = sb.matches(competition_id=9, season_id=281)
        match_row = matches[matches['match_id'] == match_id].iloc[0]
        is_home = match_row['home_team'] == team_name
        opponent = match_row['away_team'] if is_home else match_row['home_team']
        game_label = f"{opponent} ({'H' if is_home else 'A'})"
    except Exception:
        game_label = None

    events = sb.events(match_id=match_id)

    # Define Touch types
    touch_types = ['Pass', 'Carry', 'Ball Receipt*', 'Dribble', 'Shot', 'Ball Recovery']
    
    player_events = events[(events['player'] == player_name) & (events['type'].isin(touch_types))].copy()

    player_events = player_events.dropna(subset=['location'])
    player_events['x'] = player_events['location'].str[0]
    player_events['y'] = player_events['location'].str[1]

    player_events['endX'] = player_events['pass_end_location'].str[0]
    player_events['endY'] = player_events['pass_end_location'].str[1]
    
    # Normalize outcomes for coloring
    player_events['outcome'] = 'Successful'
    if 'pass_outcome' in player_events.columns:
        player_events.loc[player_events['pass_outcome'].notnull(), 'outcome'] = 'Unsuccessful'
    if 'carry_outcome' in player_events.columns:
        player_events.loc[player_events['carry_outcome'].notnull(), 'outcome'] = 'Unsuccessful'

    if 'pass_goal_assist' in player_events.columns:
        player_events.loc[player_events['pass_goal_assist'].notnull(), 'pass_goal_assist'] = 'True'
    
    print(game_label)
    stats_row = load_game_stats(game_label) if game_label else pd.Series()
    convert_to_heatmap(player_events, 'x', 'y', 'endX', 'endY', game_stats=stats_row)

if __name__ == "__main__":
    match_id = int(input("Enter Match ID: "))
    load_player_events(player_name="Florian Wirtz", match_id=match_id)