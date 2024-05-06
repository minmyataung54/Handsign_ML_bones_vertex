from scipy.spatial.distance import euclidean 
import pandas as pd
import random

df = pd.read_csv('combined_handSign.csv')

df.columns = df.columns.str.strip()

def calculate_distances(df):
    distances = {
        'wrist_thumb0': [],
        'wrist_thumb1': [],
        'wrist_thumb2': [],
        'wrist_thumb3': [],
        
        'wrist_index1': [],
        'wrist_index2': [],
        'wrist_index3': [],

        'wrist_middle1': [],
        'wrist_middle2': [],
        'wrist_middle3': [],

        'wrist_ring1': [],
        'wrist_ring2': [],
        'wrist_ring3': [],

        'wrist_pinky0': [],
        'wrist_pinky1': [],
        'wrist_pinky2': [],
        'wrist_pinky3': []
        
    }
    
    for _, row in df.iterrows():
        wrist_pos = [row['b_l_wrist_Position_X'], row['b_l_wrist_Position_Y'], row['b_l_wrist_Position_Z']]
        thumb0_pos = [row['b_l_thumb0_Position_X'], row['b_l_thumb0_Position_Y'], row['b_l_thumb0_Position_Z']]
        thumb1_pos = [row['b_l_thumb1_Position_X'], row['b_l_thumb1_Position_Y'], row['b_l_thumb1_Position_Z']]
        thumb2_pos = [row['b_l_thumb2_Position_X'], row['b_l_thumb2_Position_Y'], row['b_l_thumb2_Position_Z']]
        thumb3_pos = [row['b_l_thumb3_Position_X'], row['b_l_thumb3_Position_Y'], row['b_l_thumb3_Position_Z']]

        index1_pos= [row['b_l_index1_Position_X'], row['b_l_index1_Position_Y'], row['b_l_index1_Position_Z']]
        index2_pos= [row['b_l_index2_Position_X'], row['b_l_index2_Position_Y'], row['b_l_index2_Position_Z']]
        index3_pos= [row['b_l_index3_Position_X'], row['b_l_index3_Position_Y'], row['b_l_index3_Position_Z']]

        middle1_pos = [row['b_l_middle1_Position_X'], row['b_l_middle1_Position_Y'], row['b_l_middle1_Position_Z']]
        middle2_pos = [row['b_l_middle2_Position_X'], row['b_l_middle2_Position_Y'], row['b_l_middle2_Position_Z']]
        middle3_pos = [row['b_l_middle3_Position_X'], row['b_l_middle3_Position_Y'], row['b_l_middle3_Position_Z']]

        ring1_pos = [row['b_l_ring1_Position_X'], row['b_l_ring1_Position_Y'], row['b_l_ring1_Position_Z']]
        ring2_pos = [row['b_l_ring2_Position_X'], row['b_l_ring2_Position_Y'], row['b_l_ring2_Position_Z']]
        ring3_pos = [row['b_l_ring3_Position_X'], row['b_l_ring3_Position_Y'], row['b_l_ring3_Position_Z']]

        pinky0_pos = [row['b_l_pinky0_Position_X'], row['b_l_pinky0_Position_Y'], row['b_l_pinky0_Position_Z']]
        pinky1_pos = [row['b_l_pinky1_Position_X'], row['b_l_pinky1_Position_Y'], row['b_l_pinky1_Position_Z']]
        pinky2_pos = [row['b_l_pinky2_Position_X'], row['b_l_pinky2_Position_Y'], row['b_l_pinky2_Position_Z']]
        pinky3_pos = [row['b_l_pinky3_Position_X'], row['b_l_pinky3_Position_Y'], row['b_l_pinky3_Position_Z']]

        distances['wrist_thumb0'].append(euclidean(wrist_pos, thumb0_pos))
        distances['wrist_thumb1'].append(euclidean(wrist_pos, thumb1_pos))
        distances['wrist_thumb2'].append(euclidean(wrist_pos, thumb2_pos))
        distances['wrist_thumb3'].append(euclidean(wrist_pos, thumb3_pos))

        distances['wrist_index1'].append(euclidean(wrist_pos, index1_pos))
        distances['wrist_index2'].append(euclidean(wrist_pos, index2_pos))
        distances['wrist_index3'].append(euclidean(wrist_pos, index3_pos))

        distances['wrist_middle1'].append(euclidean(wrist_pos, middle1_pos))
        distances['wrist_middle2'].append(euclidean(wrist_pos, middle2_pos))
        distances['wrist_middle3'].append(euclidean(wrist_pos, middle3_pos))

        distances['wrist_ring1'].append(euclidean(wrist_pos, ring1_pos))
        distances['wrist_ring2'].append(euclidean(wrist_pos, ring2_pos))
        distances['wrist_ring3'].append(euclidean(wrist_pos, ring3_pos))

        distances['wrist_pinky0'].append(euclidean(wrist_pos, pinky0_pos))
        distances['wrist_pinky1'].append(euclidean(wrist_pos, pinky1_pos))
        distances['wrist_pinky2'].append(euclidean(wrist_pos, pinky2_pos))
        distances['wrist_pinky3'].append(euclidean(wrist_pos, pinky3_pos))
        
    
    return pd.DataFrame(distances)

distances_df = calculate_distances(df)

distances_df['Sign']=df['label']
shuffled_df = distances_df.sample(frac=1)

print(shuffled_df)
shuffled_df.to_csv('calculate_not_shuffle_distance_bones.csv',index=False)
print(shuffled_df.columns)
