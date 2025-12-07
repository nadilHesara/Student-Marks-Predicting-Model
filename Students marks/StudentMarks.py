import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

import matplotlib.pyplot as plt

from scipy import stats

from pathlib import Path


marks = pd.read_csv(Path("Students marks") / "Student_Marks.csv")


X = marks[['number_courses', 'time_study']]
y = marks['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


#####################Evaluating the model#######################

MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

print("MAE: ", MAE)
print("RMSE: ", RMSE)
print("R2: ", R2)


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()



##############predicted marks for some students#################

pred_inputs = pd.DataFrame(
    [[2, 5.321],
     [6, 20.534],
     [4, 7.432],
     [9, 5.0],
     [5, 9.5]],
    columns=X.columns
)

preds = model.predict(pred_inputs)
Jim, Kamal, Nimal, Haritha, Gamunu = preds

print(Jim, Kamal, Nimal, Haritha, Gamunu)

################################################################





