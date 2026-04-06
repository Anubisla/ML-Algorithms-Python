# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Simple dataset (dummy but realistic)
data = {
    "hours_study": [1, 2, 3, 4, 5, 6, 7, 8],
    "sleep_hours": [8, 7, 7, 6, 6, 5, 5, 4],
    "marks": [30, 35, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Features & target
X = df[["hours_study", "sleep_hours"]]
y = df["marks"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
