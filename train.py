import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATASET_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATASET_PATH, sep=";")

corr = data.corr()["quality"].abs()
selected = corr[corr > 0.1].index
X = data[selected].drop("quality", axis=1)
y = data["quality"]

MODEL_TYPE = "rf"
USE_SCALER = False
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump({
        "model": MODEL_TYPE,
        "features": list(X.columns),
        "test_size": TEST_SIZE,
        "mse": mse,
        "r2": r2
    }, f, indent=4)
