
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("Rows after removing duplicates:", len(concrete))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = "neg_root_mean_squared_error"


linear_model = LinearRegression()

baseline_scores = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=folds,
    scoring=scoring
)

baseline_mean = baseline_scores.mean()

print("\nLinear Regression Baseline")
print("Fold RMSE:", baseline_scores.round(2))
print("Mean CV RMSE:", round(baseline_mean, 2))

ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

parameter_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

search = GridSearchCV(
    ridge_model,
    parameter_grid,
    cv=folds,
    scoring=scoring,
    refit=True
)


search.fit(X_train, y_train)

print("\nBest Parameters:", search.best_params_)
print("Best Mean CV RMSE:", round(-search.best_score_, 2))

results = pd.DataFrame(search.cv_results_)

top_three = results.sort_values("rank_test_score").head(3)

print("\nTop Three Grid Search Settings:")
print(
    top_three[
        ["param_poly__degree", "param_ridge__alpha",
         "rank_test_score", "mean_test_score"]
    ].to_string(index=False)
)


test_predictions = search.predict(X_test)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

print("\nFinal Test Results")
print("Test RMSE:", round(test_rmse, 2))
print("Test MAE:", round(test_mae, 2))


plt.scatter(
    y_test,
    test_predictions,
    label="Predictions"
)


minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    label="Perfect prediction (y=x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.tight_layout()

plt.savefig("task3_concrete.png", dpi=300)
plt.show()


print("\nModel Comparison")
print("Baseline Mean CV RMSE:", round(baseline_mean, 2))
print("Tuned Mean CV RMSE:", round(-search.best_score_, 2))
