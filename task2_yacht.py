
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)


X = yachts[[5]]
y = yachts[6]

linear_model = LinearRegression()


quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(
    linear_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

quadratic_rmse = -cross_val_score(
    quadratic_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Display results
print("Linear Regression")
print("Fold RMSE:", linear_rmse.round(2))
print("Mean RMSE:", round(linear_rmse.mean(), 2))

print("\nQuadratic Polynomial Regression")
print("Fold RMSE:", quadratic_rmse.round(2))
print("Mean RMSE:", round(quadratic_rmse.mean(), 2))

# Compare models
if quadratic_rmse.mean() < linear_rmse.mean():
    print("\nBetter Model: Quadratic Polynomial Regression")
else:
    print("\nBetter Model: Linear Regression")
