# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the FIFA 21 player data 
fifa21_data = pd.read_csv('fifa21data/players_21.csv')
# Remove duplicate rows
fifa21_data.drop_duplicates(inplace=True)
# Drop irrelevant columns
fifa21_data.drop(['sofifa_id', 'player_url', 'long_name', 'dob', 'real_face', 'player_tags','team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until', 'nation_position'],axis=1,inplace=True)
# Fill missing values
fifa21_data['release_clause_eur'].fillna(fifa21_data['value_eur'], inplace=True)
# Visualize player ages
plt.hist(fifa21_data['age'], bins=20)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.show()
# Visualize player ratings vs wage
sns.scatterplot(x='overall', y='wage_eur', data=fifa21_data)
plt.title('Player Ratings vs. Wage')
plt.xlabel('Overall Rating')
plt.ylabel('Wage (in EUR)')
plt.show()
# Visualize top 10 countries with the most players
top_countries = fifa21_data['nationality'].value_counts().head(10)
plt.bar(top_countries.index, top_countries.values)
plt.title('Top 10 Countries with the Most Players')
plt.xlabel('Country')
plt.ylabel('Number of Players')
plt.show()
# Create new feature 'Overall_Rank'
fifa21_data['Overall_Rank'] = fifa21_data['overall'].rank(method='dense', ascending=False)
# Machine learning - Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fifa21_data[['overall', 'potential', 'age', 'international_reputation']],fifa21_data['market_value_eur'], test_size=0.3)
# Create Linear Regression model and fit it on training data
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Make predictions on testing data and evaluate performance
y_pred = linear_model.predict(X_test)
print('R-squared score: ', r2_score(y_test, y_pred))
