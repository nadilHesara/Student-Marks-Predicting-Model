import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

marks = pd.read_csv(Path("Students marks") / "Student_Marks.csv")


X = marks[['number_courses', 'time_study']]
y = marks['Marks']

regr = linear_model.LinearRegression()

regr.fit(X,y)

##############predicted marks for some students#################

pred_inputs = pd.DataFrame(
    [[2, 5.321],
     [6, 20.534],
     [4, 7.432],
     [9, 5.0],
     [5, 9.5]],
    columns=X.columns
)

preds = regr.predict(pred_inputs)
Jim, Kamal, Nimal, Haritha, Gamunu = preds

print(Jim, Kamal, Nimal, Haritha, Gamunu)

################################################################


