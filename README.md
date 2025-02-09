# Titanic Survival Prediction Model

## Overview
# Titanic Survival Prediction

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The model is built using the Titanic dataset from Kaggle and leverages classification algorithms to predict whether a passenger survived or not based on various features such as age, sex, class, and other variables.

## Table of Contents
1. [Project Overview]
2. [Dataset]
3. [Technologies Used]
4. [Model]
5. [How to Run the Project]
6. [Evaluation Metrics]
7. [Conclusion]

---

## Project Overview

In this project, we train a machine learning model to predict the survival of passengers on the Titanic based on features like age, sex, passenger class, and more. The goal is to demonstrate how various machine learning models can be used for classification tasks.

The project uses the following classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Each model is trained on the Titanic dataset, and various metrics (accuracy, precision, recall, F1 score) are used to evaluate their performance.

---

## Dataset

The dataset used in this project is the **Titanic dataset** available on Kaggle. It contains the following key columns:
- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
- **Name**: Passenger's name
- **Sex**: Gender of the passenger (male or female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Fare paid for the ticket
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: Target variable (1 = Survived, 0 = Did not survive)

---

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib/seaborn**: Visualization
- **joblib**: Model serialization (saving the trained model)

---

## Model

The following models were tested and evaluated:
1. **Logistic Regression**: A linear model for binary classification.
2. **Decision Tree Classifier**: A non-linear model based on recursive partitioning.
3. **Random Forest Classifier**: An ensemble method using multiple decision trees.
4. **Support Vector Machine (SVM)**: A model that finds the hyperplane that best separates the classes.

### Model Evaluation:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of positive predictions that are actually positive (important when false positives are costly).
- **Recall**: The proportion of actual positives that were correctly predicted (important when false negatives are costly).
- **F1 Score**: The harmonic mean of precision and recall, used as a balanced measure.

After evaluating the models, the **Random Forest** model was found to be the best performer in terms of both accuracy and precision.

---

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/titanic-prediction.git
   cd titanic-prediction

2.  Install the required dependencies:
    pip install -r requirements.txt
    pip install pandas scikit-learn joblib

3. Loading the model using joblib
   import joblib

**Load the saved model**
  model = joblib.load('titanic_model.pkl')

**Sample data for prediction**
**Format: [Pclass, Age, SibSp, Parch, Fare, Sex (0=female, 1=male), Embarked (0=C, 1=Q, 2=S)]**
  sample_data = [[3, 22, 1, 0, 7.25, 0, 0]]  # Replace this with the actual data you want to predict

**Predict survival**
  prediction = model.predict(sample_data)

**Output the prediction result**
  if prediction == 1:
      print("The passenger survived.")
  else:
      print("The passenger did not survive.")


## Evaluation Metrics
Here are the evaluation results for the models trained:

**Logistic Regression:**

Accuracy: 79%
Precision: 78%
Recall: 70%
F1 Score: 74%

**Random Forest:**

Accuracy: 81%
Precision: 83%
Recall: 68%
F1 Score: 75%

**Decision Tree:**

Accuracy: 77%
Precision: 74%
Recall: 68%
F1 Score: 71%

**Support Vector Machine (SVM):**

Accuracy: 61%
Precision: 69%
Recall: 10%
F1 Score: 17%

The **Random Forest** model is the best performing model with the highest accuracy and precision. However, recall can still be improved to ensure fewer false negatives.

## Conclusion
This project demonstrates how to use machine learning models for classification tasks, specifically predicting the survival of Titanic passengers. While the Random Forest model performs well overall, further improvements can be made by tuning hyperparameters and improving recall. This project also shows the importance of evaluating different metrics based on the context of the problem.
