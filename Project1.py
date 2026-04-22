import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv(r"C:\Users\tarun\OneDrive\Desktop\Project INT375\city_day.csv", encoding="latin-1")

print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.dropna(subset=['AQI'], inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

top_cities = df.groupby('City')['AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
top_cities.head(10).plot(kind='bar', color='tomato', edgecolor='black', linewidth=0.5)
plt.title("Top 10 Most Polluted Cities by Average AQI", fontsize=14, fontweight='bold')
plt.ylabel("Average AQI", fontsize=12)
plt.xlabel("City", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plot1_top_cities.png", bbox_inches='tight')
plt.show()

yearly_trend = df.groupby('Year')['AQI'].mean()

plt.figure(figsize=(8, 5))
yearly_trend.plot(marker='o', color='steelblue', linewidth=2, markersize=6)
plt.title("Yearly AQI Trend in India", fontsize=14, fontweight='bold')
plt.ylabel("Average AQI", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.tight_layout()
plt.savefig("plot2_yearly_trend.png", bbox_inches='tight')
plt.show()

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(8, 5))
sns.lineplot(x='Month', y='AQI', data=df, color='darkorange', linewidth=2)
plt.title("Monthly AQI Trend", fontsize=14, fontweight='bold')
plt.ylabel("Average AQI", fontsize=12)
plt.xlabel("Month", fontsize=12)
plt.xticks(ticks=range(1, 13), labels=month_names)
plt.tight_layout()
plt.savefig("plot3_monthly_trend.png", bbox_inches='tight')
plt.show()

aqi_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
palette = ['#2ecc71', '#a8e063', '#f9ca24', '#f0932b', '#e55039', '#8e1e1e']

plt.figure(figsize=(8, 5))
sns.countplot(x='AQI_Bucket', data=df, order=aqi_order, palette=palette, hue='AQI_Bucket', legend=False)
plt.title("AQI Category Distribution", fontsize=14, fontweight='bold')
plt.xlabel("AQI Category", fontsize=12)
plt.ylabel("Number of Days", fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("plot4_aqi_distribution.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, annot_kws={'size': 8})
plt.title("Correlation Between Pollutants and AQI", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot5_heatmap.png", bbox_inches='tight')
plt.show()

top_n_cities = df.groupby('City')['AQI'].median().nlargest(10).index
order = df[df['City'].isin(top_n_cities)].groupby('City')['AQI'].median().sort_values(ascending=False).index

plt.figure(figsize=(8, 7))
sns.boxplot(y='City', x='AQI', data=df[df['City'].isin(top_n_cities)],
            hue='City', palette='Reds_r', order=order, legend=False)
plt.title("Top 10 Cities – AQI Distribution", fontsize=14, fontweight='bold')
plt.xlabel("AQI", fontsize=12)
plt.ylabel("City", fontsize=12)
plt.tight_layout()
plt.savefig("plot6_citywise_box.png", bbox_inches='tight')
plt.show()

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

X = df[features].fillna(df[features].mean())
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print("\nModel Accuracy (R2 Score):", round(r2, 4))
print("Mean Absolute Error (MAE):", round(mae, 2))

importance = pd.Series(model.coef_, index=features)
print("\nPollutant Importance:\n")
print(importance.sort_values(ascending=False).round(4))

colors = ['tomato' if v >= 0 else 'steelblue' for v in importance.sort_values()]

plt.figure(figsize=(8, 6))
importance.sort_values().plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
plt.title("Pollutant Impact on AQI (Feature Importance)", fontsize=14, fontweight='bold')
plt.xlabel("Regression Coefficient", fontsize=12)
plt.ylabel("Pollutant", fontsize=12)
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig("plot7_feature_importance.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, prediction, alpha=0.4, color='steelblue', edgecolors='none', s=15)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=1.5, label='Perfect Prediction')
plt.title(f"Actual vs Predicted AQI  (R² = {round(r2, 3)})", fontsize=14, fontweight='bold')
plt.xlabel("Actual AQI", fontsize=12)
plt.ylabel("Predicted AQI", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plot8_actual_vs_predicted.png", bbox_inches='tight')
plt.show()

print("\nEDA Project Completed Successfully")