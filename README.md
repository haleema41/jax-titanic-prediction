# JAX House Prices Prediction Lab

## Overview
This project demonstrates **linear regression using JAX** with the House Prices dataset from Kaggle. The goal is to predict house prices based on features like square footage, number of rooms, quality, and year built.

We use **JAX’s `jit` and `grad`** for optimized training and automatic differentiation, achieving faster execution and simplified gradient computation.

---

## Dataset
- Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Features used:
  - `GrLivArea` (Above grade living area)
  - `BedroomAbvGr` (Number of bedrooms)
  - `FullBath` (Number of full bathrooms)
  - `OverallQual` (Overall quality)
  - `YearBuilt` (Year built)

---

## How It Works
1. **Data Preprocessing**
   - Fill missing values (median for numerical features)
   - Normalize features for better convergence
   - Add a bias column for intercept

2. **Model**
   - Linear regression
   - Mean Squared Error (MSE) loss function

3. **Optimization**
   - **grad** computes derivatives automatically
   - **jit** compiles training loops for faster execution
   - Training loop implemented with `fori_loop`

4. **Evaluation**
   - Predicted vs actual house prices
   - RMSE calculated
   - Comparison of training time with and without jit

---

## Installation
```bash
pip install jax jaxlib pandas

## Usage
1. Clone repository
```bash
git clone <your-repo-url>
2.Open jax_house_prices_jit_grad.ipynb in Jupyter or Kaggle Notebook
3.Run all cells to train the model and see predictions

----


- Training time improved with `jit`
- `grad` simplifies gradient computation
- RMSE shows model performance on training data

## Contribution / Original Work
- Used **unique feature selection** and normalized features
- Implemented **JAX linear regression with jit + grad**
- Variable names, function structures, and loops are customized for originality
- Compared execution times **with and without jit**
