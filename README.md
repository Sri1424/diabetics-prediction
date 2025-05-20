# **Diabetes Prediction using Machine Learning**
📌 **Project Description**
This project is a machine learning-based approach to predict diabetes using the Pima Indians Diabetes Dataset. The primary objective is to analyze patient medical data and classify whether an individual is diabetic or not. The project implements multiple classification algorithms, compares their performance, and outputs evaluation metrics such as accuracy, confusion matrix, and classification report.

**Dataset Overview**
Dataset Name: diabetes.csv
Source: Kaggle - Pima Indians Diabetes Dataset
Total Samples: 768

**Features:**
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (Target: 1 = Diabetic, 0 = Non-Diabetic)

🔍 **Key Steps Performed**
_1. Data Loading & Exploration_
Loaded the dataset using pandas
Displayed dataset information (.info()), summary statistics (.describe()), and checked for missing/duplicate entries
Visualized feature correlations using a heatmap

_2. Preprocessing_
Removed potential issues with LabelEncoder on numeric features (⚠️ not ideal in real scenarios)
Normalized the features using MinMaxScaler
Split the data into training and testing sets (80/20)

_3. Model Building_
Trained and evaluated four machine learning models:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (n=14)
Gaussian Naive Bayes

_4. Model Evaluation_
Used the following metrics to evaluate each model:
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
Accuracy Score

Each model was trained on the training set and tested on unseen test data.

✅**Results Summary**
Each classifier's performance is printed during execution, including:
Model name
Confusion matrix
Classification report
Accuracy score

This helps in understanding which algorithm performs best for this dataset.

**Dependencies**
Make sure the following Python libraries are installed:
pandas
numpy
matplotlib
seaborn
scikit-learn

You can install them using:
pip install pandas numpy matplotlib seaborn scikit-learn

🚀**How to Run**
Clone this repository or download the code.

Make sure the diabetes.csv file is correctly placed in your path.

_Run the Python script:_
python diabetes_prediction.py
The script will output the performance of each model.

⚠️ **Note**
Label encoding was applied to numerical features, which is not typically recommended. A better approach would be to leave continuous values as-is or use binning/discretization only if required.
MinMaxScaler was correctly applied to the training set but should use transform() on the test set instead of fit_transform() to avoid data leakage.

📜 **License**
This project is open-source and available for academic and non-commercial use.
