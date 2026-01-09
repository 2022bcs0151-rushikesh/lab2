import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATASET_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(DATASET_PATH, sep=";")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Experiment config
MODEL_TYPE = "rf"
USE_SCALER = False
TEST_SIZE = 0.2

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

# Save model
joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")

# Save results
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(
        {
            "model": MODEL_TYPE,
            "scaler": USE_SCALER,
            "test_size": TEST_SIZE,
            "mse": mse,
            "r2": r2
        },
        f,
        indent=4
    )
