# House Price Prediction Using Linear Regression

## Project Overview
This project predicts house prices using machine learning and the Linear Regression algorithm. It uses property features to estimate the sale price of a house.

## Objectives
- Perform Exploratory Data Analysis (EDA).
- Check missing values and descriptive statistics.
- Handle missing data and encode categorical features.
- Train and test a Linear Regression model.
- Evaluate the model using MSE, RMSE, and R-squared.
- Visualize actual and predicted house prices.
- Analyze residuals and feature coefficients.
- Compare Linear Regression with Ridge Regression.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset
**Dataset:** House Prices - Advanced Regression Techniques  
**Source:** Kaggle  
**Target variable:** `SalePrice`

## Project Workflow
1. Load the dataset.
2. Explore and analyze the data.
3. Handle missing values.
4. Encode categorical variables and scale numerical features.
5. Split the data into training and testing sets.
6. Train the Linear Regression model.
7. Evaluate and visualize the results.
8. Compare the model with Ridge Regression.

## How to Run the Project
1. Keep `train.csv` and `house_price.py` in the same folder.
2. Install the required libraries:

   `python -m pip install pandas numpy scikit-learn matplotlib seaborn`

3. Run the program:

   `python house_price.py`

## Output
The program displays data analysis, evaluation metrics, feature coefficients, and a model comparison. It also saves the following graphs:

- `actual_vs_predicted.png`
- `residual_plot.png`

## Conclusion
This project demonstrates how machine learning can be used to predict house prices through data analysis, preprocessing, model training, evaluation, and visualization.