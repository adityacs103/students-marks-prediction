#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df  = pd.read_csv('student_info.csv')


df2 = df.fillna(df.mean())

x = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=51)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)

lr.predict([[4]])[0][0].round(2)

y_pred  = lr.predict(x_test)

pd.DataFrame(np.c_[x_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])
lr.score(x_test,y_test)

import joblib
joblib.dump(lr, "student_mark_predictor.pkl")

model = joblib.load("student_mark_predictor.pkl")