import pandas as pd
from sklearn.preprocessing import StandardScaler

love_df = pd.read_csv('Love_sign.csv')
okay_df = pd.read_csv('Okay_sign.csv')
phone_df = pd.read_csv('phone_sign.csv')

love_df['label'] = 'love'
okay_df['label'] = 'okay'
phone_df['label'] = 'phone'

combined_df = pd.concat([love_df,okay_df,phone_df],ignore_index=True)
combined_df.to_csv('combined_handSign.csv',index=False)

# Separate features and target variable
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally convert scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)