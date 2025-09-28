🚗 Ford Used Car Price Prediction (Linear Regression Model)

Overview

This project focuses on building a robust machine learning model to predict the selling price of used Ford vehicles. The goal is to apply fundamental data analysis, visualization, and linear modeling techniques to a real-world regression problem.

The initial model successfully established a strong predictive relationship, achieving an R2 score of 0.84.

🚀 Key Results
Metric                Value                Interpretation

Model Used      Linear Regression        A foundational regression algorithm.

R2Score               0.84               84% of the variance in car prices is explained by the model's features.

Adjusted R2          0.8393               High confidence in the model's performance without overfitting.

Dataset Size      17,966 records          Large dataset providing reliable training data.


🛠️ Technology Stack

The project utilizes the core Python data science ecosystem:

Python: Primary programming language.

Pandas & NumPy: Essential for data manipulation, cleaning, and numerical operations.

Matplotlib & Seaborn: Used for generating informative visualizations (histograms, boxplots, correlation heatmaps).

Scikit-learn: Used for implementing the Linear Regression model and calculating performance metrics.


📊 Data Analysis & Methodology

The core steps followed in the Jupyter Notebook (car_price.ipynb):

Exploratory Data Analysis (EDA):
Checked for missing values (none found) and analyzed data types.
Visualized key relationships, such as the correlation between mileage and price, and the distribution of cars by year.

Feature Engineering & Preprocessing:
Handled categorical variables (model, transmission, fuelType) using One-Hot Encoding to convert them into a machine-readable format.
Split the data into training and testing sets (75/25 split).

Model Training:
A Linear Regression model was trained on the preprocessed feature set.

Evaluation:
The model was evaluated using standard regression metrics, confirming the final R2 score of 0.84.

🔮 Future Improvements (Next Steps)
This project serves as a strong baseline. To significantly push the predictive accuracy further (aiming for R 
2>0.90), the following steps are planned:

Advanced Feature Engineering: Create the new feature car_age from the year column, which is often a more powerful predictor of price.

Outlier Handling: Systematically detect and manage influential outliers identified during the EDA phase.

Model Comparison: Implement and compare performance against more sophisticated models better suited for non-linear data, such as Random Forest Regressor and Gradient Boosting Regressor (requires Scikit-learn).

⚙️ How to Run the Project Locally
Clone the repository:

git clone [https://github.com/Gaurav-Hirani/Car-Price-Prediction-ML.git](https://github.com/Gaurav-Hirani/Car-Price-Prediction-ML.git)
cd Car-Price-Prediction-ML

Install dependencies: Ensure you have Python installed, then install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn
