# Calorie Burn Prediction

## Overview
This project aims to predict the number of calories burned based on exercise data using machine learning models. The dataset contains user-specific details such as age, gender, duration of exercise, and other features to train regression models for accurate predictions.

## Dataset
- The dataset consists of two CSV files:
  - **calories.csv**: Contains calorie burn information.
  - **exercise.csv**: Contains exercise-related features like duration, age, and gender.
- After merging both datasets, preprocessing steps include handling missing values, removing duplicates, encoding categorical variables, and normalizing numerical features.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Plotly

## Preprocessing Steps
1. Merged `exercise.csv` and `calories.csv` on `User_ID`.
2. Removed duplicate entries based on `User_ID`.
3. Encoded gender (`male = 0`, `female = 1`).
4. Dropped unnecessary columns (`User_ID`).
5. Performed correlation analysis to identify feature importance.

## Model Training
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- The dataset was split into training (80%) and testing (20%).
- Model evaluation used Mean Squared Error (MSE) and R² score.

## Results
- The trained models achieved high accuracy, with Random Forest and XGBoost performing the best.
- The R² score for training and testing data indicates the model's effectiveness in predicting calorie burn.

## Visualization
- **Scatter Plot**: Shows the relationship between exercise duration and calories burned.
- **Distribution Plot**: Displays the age distribution in the dataset.
- **Heatmap**: Visualizes feature correlations to understand their impact on calorie burn.

## Usage
To use the model for predictions:
1. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly
   ```
2. Run the script to preprocess data and train the model.
3. Input new exercise data to get calorie burn predictions.

## Future Improvements
- Enhance feature engineering to include additional activity parameters.
- Experiment with deep learning models for better accuracy.
- Deploy the model as a web application for real-time predictions.

---


