# California House Price Prediction

This project uses machine learning to predict housing prices in California based on various features. The data is sourced from the California Housing dataset, which provides insights into the housing market in different regions of California.

## Project Overview

This project was inspired by my studies while reading Hands-on Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron. Drawing on the concepts and techniques from the book, I have implemented this project to predict median house prices in California districts using various machine learning algorithms and evaluate their performance.

## Dataset

The dataset contains information on various features of the housing districts, including:

- **Median Income**: The median income of households in a district.
- **Housing Median Age**: The median age of houses in a district.
- **Total Rooms**: The total number of rooms in a district.
- **Total Bedrooms**: The total number of bedrooms in a district.
- **Population**: The population of the district.
- **Households**: The number of households in a district.
- **Latitude**: The latitude of the district.
- **Longitude**: The longitude of the district.

## Project Structure

- **Data Preprocessing**: The data is cleaned, and missing values are handled. Feature scaling and transformation are applied where necessary.
- **Exploratory Data Analysis (EDA)**: Visualization techniques are used to understand the data distribution, correlations, and outliers.
- **Modeling**: Several machine learning models are applied, including:
  - Linear Regression
  - Decision Trees
  - Random Forest

- **Model Evaluation**: The models are evaluated using metrics like Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- **Hyperparameter Tuning**: Grid Search and Randomized Search are used to fine-tune the models for optimal performance.

## Key Results

- The **Random Forest Regressor** performed the best, achieving the lowest RMSE on the test data.
- **Feature Importance** analysis showed that the median income was the most significant predictor of housing prices.

## Tools and Libraries

- **Python**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/VyshnaviKurivella/California-House-Price-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd California-House-Price-Prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
5. Open and run the notebook `California_Housing_Prediction.ipynb`.

## Future Work

- Experiment with other advanced machine learning models like XGBoost or Neural Networks.
- Perform more feature engineering to improve model performance.
- Deploy the model as a web application.
