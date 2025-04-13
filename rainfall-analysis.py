import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, silhouette_score

# Load dataset
data=pd.read_csv("C:\\Users\\priya\\Downloads\\rainfall in india 1901-2015.csv")
# Preview data
print(data.head())
print(data.info())
print(data.describe())
print(data.isna().sum())
# Handle missing values (fill NA with mean)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Grouping by 'SUBDIVISION' and 'YEAR' for regional analysis
grouped = data.groupby(['SUBDIVISION', 'YEAR']).mean().reset_index()

# ---------------- EDA 1: Trend of Rainfall Over the Years (Nationwide) ----------------
nationwide = data.groupby('YEAR')['ANNUAL'].mean()
plt.figure(figsize=(10,5))
plt.plot(nationwide.index, nationwide.values, color='blue')
plt.title("Trend of Annual Rainfall in India (1901–2015)")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall (mm)")
plt.grid(True)
plt.show()

# ---------------- EDA 2: Top 5 Wettest and Driest Subdivisions ----------------
avg_rain = data.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
avg_rain.head(5).plot(kind='bar', color='teal')
plt.title("Top 5 Wettest Regions")
plt.ylabel("Average Annual Rainfall (mm)")
plt.show()
#and
plt.figure(figsize=(10,5))
avg_rain.tail(5).plot(kind='bar', color='coral')
plt.title("Top 5 Driest Regions")
plt.ylabel("Average Annual Rainfall (mm)")
plt.show()

# ---------------- EDA 3: Seasonal Contribution to Annual Rainfall ----------------
seasonal = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()
seasons = {
    'Winter': seasonal[['JAN', 'FEB']].sum(),
    'Summer': seasonal[['MAR', 'APR', 'MAY']].sum(),
    'Monsoon': seasonal[['JUN', 'JUL', 'AUG', 'SEP']].sum(),
    'Post-Monsoon': seasonal[['OCT', 'NOV', 'DEC']].sum()
}
plt.figure(figsize=(8,5))
plt.pie(seasons.values(), labels=seasons.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Seasonal Contribution to Annual Rainfall")
plt.show()

# ---------------- EDA 4: Year with Highest & Lowest Rainfall per Region ----------------
highest = data.loc[data.groupby('SUBDIVISION')['ANNUAL'].idxmax()][['SUBDIVISION', 'YEAR', 'ANNUAL']]
lowest = data.loc[data.groupby('SUBDIVISION')['ANNUAL'].idxmin()][['SUBDIVISION', 'YEAR', 'ANNUAL']]
print("\nYear with Highest Rainfall per Region:\n", highest.head())
print("\nYear with Lowest Rainfall per Region:\n", lowest.head())

# ---------------- EDA 5: Heatmap of Monthly Rainfall Correlation ----------------
plt.figure(figsize=(10,7))
corr = data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Heatmap of Monthly Rainfall Correlation")
plt.show()

# ---------------- Classification Model: Logistic Regression ----------------

# Create binary label: 1 = High Rainfall, 0 = Low Rainfall based on median
data['RainfallLevel'] = (data['ANNUAL'] > data['ANNUAL'].median()).astype(int)

features = ['JUN', 'JUL', 'AUG', 'SEP']  # Major monsoon months
X = data[features]
y = data['RainfallLevel']

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Evaluation
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", conf_mat)

# Plot Confusion Matrix

plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Rainfall', 'High Rainfall'], yticklabels=['Low Rainfall', 'High Rainfall'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# ---------------- Clustering Model: KMeans ----------------

kmeans_features = ['JUN', 'JUL', 'AUG', 'SEP', 'OCT']
X_cluster = scaler.fit_transform(data[kmeans_features])

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)
data['Cluster'] = clusters

# Silhouette Score
sil_score = silhouette_score(X_cluster, clusters)
print("\n--- KMeans Clustering ---")
print("Silhouette Score:", sil_score)

# Cluster Visualization
plt.scatter(data['JUL'], data['SEP'], c=clusters, cmap='viridis')
plt.xlabel('July Rainfall (mm)')
plt.ylabel('September Rainfall (mm)')
plt.title('KMeans Clustering Based on Monsoon Rainfall')
plt.grid(True)
plt.show()
