import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# -------------------------------
# Helper: Remove Outliers
# -------------------------------
def remove_feature_outliers(df, features, iqr_multiplier=1.5):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]
    return df

# -------------------------------
# Load and Clean Data
# -------------------------------
df = pd.read_csv("static/house_prices.csv")

# Drop irrelevant column
df.drop(columns=["index"], inplace=True, errors="ignore")

# Remove invalid values
df = df[df["Area_in_Marla"] > 0]
df.dropna(inplace=True)

# Remove numeric outliers
numeric_cols = ['Area_in_Marla', 'bedrooms', 'baths', 'price']
df = remove_feature_outliers(df, numeric_cols, iqr_multiplier=2.0)

# -------------------------------
# Create city_location_grouped Feature
# -------------------------------
df["city_location"] = df["city"].str.strip().str.lower() + "___" + df["location"].str.strip().str.lower()

top_city_locations = df["city_location"].value_counts().nlargest(100).index
df["city_location_grouped"] = df["city_location"].apply(
    lambda x: x if x in top_city_locations else "other" )

top_city_location_list = list(top_city_locations)

# writing on .json file
with open("static/city_location_list.json", "w") as f:
    json.dump(top_city_location_list, f)

print(f"Saved static/city_location_list.json with {len(top_city_location_list)} values.")

# -------------------------------
# Feature Engineering
# -------------------------------
df["log_price"] = np.log1p(df["price"])
df["Total_Rooms"] = df["bedrooms"] + df["baths"]
df["log_area"] = np.log1p(df["Area_in_Marla"])
df["area_per_room"] = df["Area_in_Marla"] / (df["Total_Rooms"] + 1)

# Drop unused features
df.drop(columns=["Area_in_Marla", "PricePerMarla"], inplace=True, errors="ignore")

# -------------------------------
# Model Preparation
# -------------------------------
cat_cols = ["city_location_grouped", "property_type", "purpose"]
num_cols = ["Total_Rooms", "log_area"]

X = df[cat_cols + num_cols]
y = df["log_price"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

# -------------------------------
# Train/Test Split and Model Training
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred_log = pipeline.predict(X_test)
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

r2_log = r2_score(y_test, y_pred_log)
r2_original = r2_score(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)

print("\nModel Evaluation:")
print(f"R² Score (log space)     : {r2_log:.4f}")
print(f"R² Score (original space): {r2_original:.4f}")
print(f"RMSE                     : {rmse:,.2f}")
print(f"MAE                      : {mae:,.2f}")

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(pipeline, "static/price_model.pkl")
print("\n Model saved to price_model.pkl")
