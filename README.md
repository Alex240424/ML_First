# Machine Learning Assignment - Polynomial Regression

**Student:** Boyina Saketh  
**Roll Number:** BT2024085  
**Institute:** International Institute of Information Technology Bangalore (IIIT-B)

## Overview

This repository contains the implementation of the Polynomial Regression
assignment for two different datasets, var1 and var2.

The objective is to build polynomial regression models, select suitable
polynomial degrees using 5-fold cross-validation, and evaluate the models
using Mean Squared Error (MSE) and R² score.

## Problems

### Problem 1 - var1

- Number of features: 6
- Polynomial degrees tested: 1 to 10
- Ridge Regression is used for regularization.
- 5-fold cross-validation is used for model selection.
- Evaluation metrics: MSE and R².

### Problem 2 - var2

- Number of features: 3
- Polynomial degrees tested: 1 to 20
- Ridge Regression is used for regularization.
- 5-fold cross-validation is used for model selection.
- Evaluation metrics: MSE and R².

## Files

| File | Description |
|---|---|
| `Phase1.py` | Training and inference code for Problem 1 (var1) |
| `Phase2.py` | Training and inference code for Problem 2 (var2) |
| `Phase1.ipynb` | Google Colab notebook for Problem 1 |
| `Phase2.ipynb` | Google Colab notebook for Problem 2 |
| `BT2024085_pred_var1.csv` | Prediction file for Problem 1 |
| `BT2024085_pred_var2.csv` | Prediction file for Problem 2 |

## Methods Used

- Polynomial Feature Generation
- Ridge Regression
- 5-Fold Cross-Validation
- Mean Squared Error (MSE)
- R² Score
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Selection

Different polynomial degrees and Ridge regularization parameters
($\alpha$) are evaluated using 5-fold cross-validation.

For each polynomial degree, the Ridge regularization parameter producing
the lowest mean cross-validation MSE is selected.

The polynomial degree with the lowest cross-validation MSE is then selected
as the final model.

## Implementation

The `.py` files contain the actual Python implementation used for training
the models and generating predictions.

The `.ipynb` files are Google Colab notebooks containing the corresponding
notebook-based implementation and analysis.

## Prediction Files

The final predictions for the two test datasets are provided as:

- `BT2024085_pred_var1.csv`
- `BT2024085_pred_var2.csv`

## Repository

GitHub Repository:

https://github.com/Alex240424/ML_First

## Author

**Boyina Saketh**  
**BT2024085**
