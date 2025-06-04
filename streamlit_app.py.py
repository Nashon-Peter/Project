import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Climate Change Analysis in Tanzania")

# Load the CSV file
df = pd.read_csv("tanzania_climate_data.csv")

st.subheader("Preview of Climate Data")
st.dataframe(df.head(18))

# Preprocessing
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Descriptive Statistics")
st.write(df.describe())

# Line plot - Temperature
st.subheader("Average Temperature Over Time")
fig, ax = plt.subplots()
ax.plot(df.index.year, df['Average_Temperature_C'], label='Avg Temp (°C)', color='green')
ax.set_title('Climate Trends in Tanzania (1980–2025)')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (°C)')
ax.legend()
st.pyplot(fig)

# Line plot - Precipitation
st.subheader("Annual Precipitation Over Time")
fig, ax = plt.subplots()
ax.plot(df.index.year, df['Annual_Precipitation_mm'], label='Precipitation (mm)', color='blue')
ax.set_title('Precipitation Trends (1980–2025)')
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation (mm)')
ax.legend()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Matrix")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)







