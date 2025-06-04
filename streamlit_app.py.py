# Climate Change Analysis in Tanzania

#  1. Install & Import Necessary Libraries
#get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn --quiet')


# Import Necessary Libraries
import pandas as pd # load data
import numpy as np # for numeric Processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns # Visualization
from sklearn.model_selection import train_test_split # Machine Learning model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose # plot time series data


# Generating mock climate data from 1980 to 2025
np.random.seed(42)
years = np.arange(1980, 2025)
temperature = 22 + 0.03 * (years - 1980) + np.random.normal(0, 0.5, len(years))
precipitation = 800 + 2 * (years - 1980) + np.random.normal(0, 30, len(years))

df = pd.DataFrame({
    'Year': years,
    'Average_Temperature_C': temperature,
    'Annual_Precipitation_mm': precipitation
})

# Save mock data to CSV
df.to_csv('tanzania_climate_data.csv', index=False)


# 3. Load and Preview the Dataset
df = pd.read_csv('tanzania_climate_data.csv')
print(df.head(18))


#  4. Data Preprocessing
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())


#  5. Exploratory Data Analysis (EDA)

# Descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())


#  5. Exploratory Data Analysis (EDA)

# Descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())

# Line plots
plt.figure(figsize=(12, 6))
plt.plot(df.index.year, df['Average_Temperature_C'], label='Avg Temp (°C)', color='green')
plt.title('Climate Trends in Tanzania (1980–2025)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()


# Line plots
plt.figure(figsize=(12, 6))

plt.plot(df.index.year, df['Annual_Precipitation_mm'], label='Precipitation (mm)', color='blue')
plt.title('Climate Trends in Tanzania (1980–2025)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()


# Heatmap for correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Seasonal decomposition (on temperature)
result = seasonal_decompose(df['Average_Temperature_C'], model='additive', period=5)
result.plot()
plt.suptitle("Seasonal Decomposition of Avg Temperature")
plt.show()


# 6. Machine Learning Modeling

# Use year as numeric feature
df_ml = df.copy()
df_ml['Year'] = df_ml.index.year


X = df_ml[['Year']]
y_temp = df_ml['Average_Temperature_C']
y_prec = df_ml['Annual_Precipitation_mm']

# Split data
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)


# Linear Regression Model for Temperature
model_1 = LinearRegression()
model_1.fit(X_train, y_temp_train)
temp_pred = model_1.predict(X_test)


#  7. Evaluation
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} Evaluation:")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - MAE: {mae:.2f}")
    print()


evaluate_model(y_temp_test, temp_pred, "Linear Regression")


# Random Forest for comparison
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_temp_train)
rf_pred = rf_model.predict(X_test)



evaluate_model(y_temp_test, temp_pred, "Linear Regression")
evaluate_model(y_temp_test, rf_pred, "Random Forest Regressor")


#  8. Predict Future Climate (2021–2030)
future_years = pd.DataFrame({'Year': np.arange(2026, 2031)})
future_temp_pred = rf_model.predict(future_years)
future_temp_pred


#  Plotting Predictions
plt.figure(figsize=(10, 5))
plt.plot(df.index.year, df['Average_Temperature_C'], label='Historical Temp', color='green')
plt.plot(future_years['Year'], future_temp_pred, label='Predicted Temp (2025-2030)', color='orange')
plt.title("Temperature Forecast for Tanzania")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.show()







