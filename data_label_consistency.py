import pandas as pd

# Load the data
love_df = pd.read_csv('Love_sign.csv')
okay_df = pd.read_csv('Okay_sign.csv')
phone_df = pd.read_csv('phone_sign.csv')

# Check for missing values and data types
print(love_df.info())  # Summarize data including null counts and data types
print(love_df.describe())  # Get descriptive statistics for numerical columns

# Label the data
love_df['label'] = 'love'
okay_df['label'] = 'okay'
phone_df['label'] = 'phone'

# Example of visualizing data
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize distributions of features
sns.pairplot(love_df.drop('label', axis=1))
plt.show()

# Check feature correlation
correlation_matrix = love_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
