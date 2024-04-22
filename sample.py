from scipy.spatial.distance import euclidean
import pandas as pd

df = pd.read_csv('combined_handSign.csv')
df.columns = df.columns.str.strip()

def calculate_distances(df, rows_to_use):
    distances = {
        'wrist_thumb0': []
        
    }
    for idx in rows_to_use:
        row = df.iloc[idx]
        wrist_pos = [row['b_l_wrist_Position_X'], row['b_l_wrist_Position_Y'], row['b_l_wrist_Position_Z']]
        thumb0_pos = [row['b_l_thumb0_Position_X'], row['b_l_thumb0_Position_Y'], row['b_l_thumb0_Position_Z']]
        print(wrist_pos)
        print(thumb0_pos)
        
        
        distances['wrist_thumb0'].append(euclidean(wrist_pos, thumb0_pos))
    return pd.DataFrame(distances)

rows_to_use = [0, 1, 2]
distances_df = calculate_distances(df, rows_to_use)


distances_df['Sign'] = df['label'].iloc[rows_to_use].values
print(distances_df)
