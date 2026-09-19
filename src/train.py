from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(7); n = 1500
df = pd.DataFrame({"area_sqft": rng.normal(1500, 500, n).clip(450), "bedrooms": rng.integers(1, 6, n), "age_years": rng.integers(0, 45, n), "neighborhood": rng.choice(["Central", "North", "East", "West"], n)})
neighborhood_value = df.neighborhood.map({"Central": 95000, "North": 60000, "East": 25000, "West": 40000})
df["price"] = 55000 + df.area_sqft * 175 + df.bedrooms * 11000 - df.age_years * 1800 + neighborhood_value + rng.normal(0, 30000, n)
X, y = df.drop(columns="price"), df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
prep = ColumnTransformer([("numeric", StandardScaler(), ["area_sqft", "bedrooms", "age_years"]), ("category", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"])])
pipeline = Pipeline([("features", prep), ("model", LinearRegression())]).fit(X_train, y_train)
pred = pipeline.predict(X_test)
root = Path(__file__).resolve().parents[1]; out = root / "artifacts"; out.mkdir(exist_ok=True)
joblib.dump(pipeline, out / "house_price_pipeline.joblib")
(out / "metrics.txt").write_text(f"RMSE: {mean_squared_error(y_test, pred) ** .5:,.0f}\nR-squared: {r2_score(y_test, pred):.3f}\n")
df.to_csv(out / "synthetic_housing_data.csv", index=False)
print((out / "metrics.txt").read_text())
