
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


homes = pd.read_csv("real_estate.csv")


homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

model = LinearRegression()
model.fit(X, y)

# Display slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


new_home = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_home)

print("Predicted price at 500 metres:", prediction[0])

line_predictions = model.predict(X)

plt.scatter(X["distance_100m"], y, label="Actual observations")
plt.plot(
    X["distance_100m"],
    line_predictions,
    color="red",
    label="Regression line"
)

plt.xlabel("Distance to nearest metro station (100 metres)")
plt.ylabel("House price per unit area")
plt.title("Housing Price Prediction using Linear Regression")
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("task1_housing.png", dpi=300)
plt.show()
