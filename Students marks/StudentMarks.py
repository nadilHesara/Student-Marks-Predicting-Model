import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats

marks = pd.read_csv("Students marks\Student_Marks.csv")


X = marks[['number_courses', 'time_study']]
y = marks['Marks']

regr = linear_model.LinearRegression()

regr.fit(X,y)

##############predicted marks for some students#################

Jim = regr.predict([[2, 5.321]])
Kamal = regr.predict([[6, 20.534]])
Nimal = regr.predict([[4, 7.432]])
Haritha = regr.predict([[9, 5]])
Gamunu = regr.predict([[5, 9.500]])

print(Jim, Kamal, Nimal, Haritha, Gamunu)

################################################################


