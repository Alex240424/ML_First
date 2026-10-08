import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Load data
train_df = pd.read_csv("BT2024085_train_var2.csv")
test_df = pd.read_csv("BT2024085_test_var2.csv")

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Separate features and target
features = ["x1", "x2", "x3"]
X = train_df[features].values
y = train_df["y"].values
X_test = test_df[features].values


# Generate exponent combinations
def generate_exponents(degree, n_features=3):
    exponents = []
    for i in range(degree + 1):
        for j in range(degree - i + 1):
            k = degree - i - j
            exponents.append((i, j, k))
    return exponents
# Create polynomial features manually
def make_polynomial_features(X, degree):
    polynomial_terms = []
    for current_degree in range(1, degree + 1):
        exponents = generate_exponents(
            current_degree,
            X.shape[1]
        )
        for powers in exponents:
            term = np.ones(X.shape[0])
            for feature_index in range(X.shape[1]):

                term = term * (
                    X[:, feature_index] ** powers[feature_index]
                )

            polynomial_terms.append(term)

    return np.column_stack(polynomial_terms)


# Calculate Ridge weights
def calculate_ridge_weights(X, y, alpha):
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    n_features = X_bias.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    weights = np.linalg.solve(X_bias.T @ X_bias + alpha * I,X_bias.T @ y)
    return weights
# Make predictions
def predict(X, weights):
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    return X_bias @ weights

# Calculate MSE
def calculate_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Calculate R2
def calculate_r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


# Setup 5-fold cross validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Alpha values to test
alphas = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
# Test degrees from 1 to 20
results = []

for degree in range(1, 21):
    best_degree_mse = float("inf")
    best_degree_r2 = None
    best_degree_alpha = None
    # Try different alpha values
    for alpha in alphas:
        mse_values = []
        r2_values = []
        # Perform 5-fold cross validation
        for train_index, val_index in kf.split(X):
            # Split training and validation data
            X_train = X[train_index]
            y_train = y[train_index]
            X_val = X[val_index]
            y_val = y[val_index]
            # Create polynomial features
            X_train_poly = make_polynomial_features(X_train,degree)
            X_val_poly = make_polynomial_features(X_val,degree)
            # Calculate Ridge weights
            weights = calculate_ridge_weights(X_train_poly,y_train,alpha)
            # Predict validation data
            y_val_pred = predict(X_val_poly,weights)
            # Calculate validation metrics
            fold_mse = calculate_mse(y_val,y_val_pred)
            fold_r2 = calculate_r2(y_val,y_val_pred)
            mse_values.append(fold_mse)
            r2_values.append(fold_r2)

        # Average CV metrics
        mean_mse = np.mean(mse_values)
        mean_r2 = np.mean(r2_values)
        # Keep best alpha for this degree
        if mean_mse < best_degree_mse:
            best_degree_mse = mean_mse
            best_degree_r2 = mean_r2
            best_degree_alpha = alpha
    # Store best result for this degree
    results.append((degree,best_degree_mse,best_degree_r2,best_degree_alpha))
    # Print CV results
    print(f"Degree {degree:2d} | "f"CV MSE: {best_degree_mse:.4f} "f"(R2: {best_degree_r2:.4f}) | "f"Alpha: {best_degree_alpha:.0e}")


# Plot CV MSE and CV R2
degrees = [r[0] for r in results]
cv_mse = [r[1] for r in results]
cv_r2 = [r[2] for r in results]


# CV MSE graph
plt.figure()
plt.plot(
    degrees,
    cv_mse,
    marker="o"
)
plt.xlabel("Polynomial Degree")
plt.ylabel("CV MSE")
plt.title("CV MSE vs Polynomial Degree")
plt.xticks(degrees)
plt.grid(True)
plt.show()


# CV R2 graph
plt.figure()
plt.plot(
    degrees,
    cv_r2,
    marker="o"
)
plt.xlabel("Polynomial Degree")
plt.ylabel("CV R2")
plt.title("CV R2 vs Polynomial Degree")
plt.xticks(degrees)
plt.grid(True)
plt.show()


# Select best degree and alpha
best_degree, best_mse, best_r2, best_alpha = min(results,key=lambda x: x[1])
print("\nBest Degree:", best_degree)
print("Best CV MSE:", best_mse)
print("Best CV R2:", best_r2)
print("Best Alpha:", best_alpha)

# Train final model using all training data
X_poly = make_polynomial_features(X,best_degree)
X_test_poly = make_polynomial_features(X_test,best_degree)

# Calculate final Ridge weights
weights = calculate_ridge_weights(X_poly,y,best_alpha)

# Calculate final training performance
y_train_pred = predict(X_poly,weights)
train_mse = calculate_mse(y,y_train_pred)

train_r2 = calculate_r2(y,y_train_pred)

print("\nFinal Training MSE:", train_mse)
print("Final Training R2:", train_r2)

# Predict test data
y_test_pred = predict(X_test_poly,weights)
# Save predictions
submission = pd.DataFrame({"y": y_test_pred})
submission.to_csv("BT2024085_predictions_var2.csv",index=False)
print("\nPredictions saved to:")
print("BT2024085_predictions_var2.csv")