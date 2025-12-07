# 📘 Student Marks Predictor

A simple yet complete machine learning project that predicts a student’s exam marks based on:

- Number of courses enrolled
- Daily study time (hours)

This project demonstrates the full ML workflow using Python, Pandas, Matplotlib, and Scikit-Learn.  
Designed to be clear, beginner-friendly, and suitable for an undergraduate ML portfolio.

---

## 🚀 Project Highlights

- End-to-end machine learning pipeline  
- Clean Exploratory Data Analysis
- Train/Test split  
- Linear Regression model  
- Evaluation metrics: MAE, RMSE, R²  
- Prediction for new student inputs  
- Reproducible Jupyter Notebook  

---

## 📂 Dataset Overview

The dataset used is:
Student_Marks.csv

Downloaded from Kaggle Datasets : https://www.kaggle.com/datasets/yasserh/student-marks-dataset


This includes:

| Column           | Description                                         |
|------------------|-----------------------------------------------------|
| number_courses   | Number of courses a student is enrolled in          |
| time_study       | Daily study time in hours                           |
| Marks            | Final exam score of the student                     |

This simple structure makes the dataset ideal for learning regression modeling.

---

## 🧪 Modeling Approach

### 1️⃣ Feature Selection  
The model uses two numerical features:
number_courses
time_study


### 2️⃣ Algorithm  
A **Linear Regression** model from scikit-learn is used because:

- It is easy to interpret  
- Performs well on small numeric datasets  

### 3️⃣ Train/Test Split  
A train-test split (80% training, 20% testing) is used to measure generalization.

### 4️⃣ Evaluation Metrics  
The model is evaluated using:

MAE: 3.0793452296666843

RMSE: 3.768385083344663

R² Score: 0.9459936100591213


---

## 📊 Visualizations Included

The notebook includes:

- Distribution of marks
- Study time vs marks scatter plot
- Number of courses vs marks
- Actual vs predicted marks

Example:

```python
plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
````

---
## 🔮 Future Improvements

- Add polynomial regression
- Use Random Forest Regressor
- Add more features to the dataset
- Add model comparison section








