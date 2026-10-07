# Machine Learning Assignment - Polynomial Regression

**Student:** Boyina Saketh  
**Roll Number:** BT2024085  
**Institute:** International Institute of Information Technology Bangalore (IIIT-B)

## Overview

This repository contains the implementation of the Polynomial Regression assignment.

The assignment involves building polynomial regression models, selecting suitable polynomial degrees using 5-fold cross-validation, and evaluating the models using Mean Squared Error (MSE) and R² score.

## Problems

### Problem 1 - var1

- Number of features: 6
- Polynomial degrees tested: 1 to 10
- Ridge Regression is used for regularization.
- 5-fold cross-validation is used for model selection.
- Evaluation metrics: MSE and R².

### Problem 2 - var2

- Number of features: 3
- Polynomial degrees tested up to 20.
- Ridge Regression is used for regularization.
- 5-fold cross-validation is used for model selection.
- Evaluation metrics: MSE and R².

## Files

| File | Description |
|---|---|
| `Phase1.ipynb` | Implementation for Problem 1 |
| `Phase2.ipynb` | Implementation for Problem 2 |
| `BT2024085_predictions_var1.csv` | Test predictions for Problem 1 |
| `BT2024085_predictions_var2.csv` | Test predictions for Problem 2 |

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

Different polynomial degrees and Ridge regularization parameters (alpha) are evaluated using 5-fold cross-validation. The model with the lowest cross-validation MSE is selected as the final model.
