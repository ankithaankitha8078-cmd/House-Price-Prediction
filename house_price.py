
# TASK 1: Predicting House Prices with Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("train.csv")

print("=" * 50)
print("HOUSE PRICE PREDICTION - DATA ANALYSIS")
print("=" * 50)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# --------------------------------------------------
# 2. EXPLORE THE DATA
# --------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDescriptive statistics:")
print(df.describe())

print("\nSale price statistics:")
print(df["SalePrice"].describe())

# Plot the target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------

# SalePrice is the variable we want to predict.
# Id is excluded because it is only a record identifier.
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

print("\nTarget variable: SalePrice")
print("Number of input features:", X.shape[1])

# Features such as quality, living area, neighborhood,
# garage capacity and year built may influence price.
# We retain the available house features for modelling.

# --------------------------------------------------
# 4. CORRELATION HEATMAP
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

correlations = numeric_df.corr()["SalePrice"].drop("SalePrice")
top_features = correlations.abs().sort_values(
    ascending=False
).head(10).index.tolist()

heatmap_data = df[top_features + ["SalePrice"]].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Top Numeric Features")
plt.tight_layout()
plt.show()

print("\nTop numerical correlations with SalePrice:")
print(correlations.loc[top_features].sort_values(
    ascending=False
))

# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 6. TRAIN/TEST SPLIT (80/20)
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 7. TRAIN LINEAR REGRESSION
# --------------------------------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

predictions = linear_model.predict(X_test)

# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)

print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

# --------------------------------------------------
# 9. ACTUAL VS PREDICTED PRICE PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot([lower, upper], [lower, upper], "r--")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# --------------------------------------------------
# 10. RESIDUAL PLOT
# --------------------------------------------------

residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted House Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.show()

# --------------------------------------------------
# 11. COEFFICIENT ANALYSIS
# --------------------------------------------------

feature_names = linear_model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = linear_model.named_steps["model"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print("\nTop 10 positive coefficients:")
print(
    coef_df.sort_values("Coefficient", ascending=False)
    .head(10).to_string(index=False)
)

print("\nTop 10 negative coefficients:")
print(
    coef_df.sort_values("Coefficient", ascending=True)
    .head(10).to_string(index=False)
)

# --------------------------------------------------
# 12. RIDGE REGRESSION COMPARISON
# --------------------------------------------------

ridge_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=10.0))
])

ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\n" + "=" * 50)
print("RIDGE REGRESSION RESULTS")
print("=" * 50)

print(f"Ridge MSE: {ridge_mse:,.2f}")
print(f"Ridge RMSE: {ridge_rmse:,.2f}")
print(f"Ridge R2: {ridge_r2:.4f}")

print("\nMODEL COMPARISON")
print("-" * 45)
print(f"{'Model':<22}{'RMSE':>13}{'R2':>10}")
print("-" * 45)
print(f"{'Linear Regression':<22}{rmse:>13,.2f}{r2:>10.4f}")
print(f"{'Ridge Regression':<22}{ridge_rmse:>13,.2f}{ridge_r2:>10.4f}")

print("\nAnalysis completed successfully!")
