import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data for AGV routes
data = {
    'distance': [10, 20, 30, 40, 50],
    'time': [5, 10, 15, 20, 25],
    'energy_consumption': [2, 4, 6, 8, 10]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Train a simple linear regression model
X = df[['distance', 'time']]
y = df['energy_consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("AGV Route Optimization")
st.write("Optimize AGV routes in real-time.")

# User input for distance and time
distance = st.number_input("Enter distance (in meters):", min_value=0)
time = st.number_input("Enter time (in minutes):", min_value=0)

# Predict energy consumption
if st.button("Optimize Route"):
    input_data = np.array([[distance, time]])
    predicted_energy = model.predict(input_data)
    st.write(f"Predicted energy consumption: {predicted_energy[0]:.2f} units")
