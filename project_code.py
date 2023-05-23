
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#data understanding
pa=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\ml project on sales _status\a.1ans.csv")
print(pa.info())
print(pa.describe())
print(pa.tail(5))
print(pa.head(5))

#data pre_processing
print(pa.isnull().sum())

x = pa.iloc[:,:-1].values
y = pa.iloc[:,-1].values
print(x)
print(y)
#train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sa = StandardScaler()
x_train = sa.fit_transform(x_train)
x_test=sa.transform(x_test)
print(x_train)

#modling
model = LogisticRegression()
model.fit(x_train,y_train)

#predict
pred = model.predict(x_test)

#fature predict
age=int(input('enter your age:-'))
salaries=int(input('enter your salaries:-'))
new_cost=[[age,salaries]]
result=model.predict(sa.transform(new_cost))
print("_______________________________________")
print(result)
if result ==1:
    print('costamer will buy')
else:
    print('not buy')
print("_______________________________________")
#result
print('accuracy_score of this is: {}'.format(accuracy_score(y_test,pred)*100))
print('confusion_matrix of this is: {}'.format(confusion_matrix(y_test,pred)))
print('classification_report of this is: {}'.format(classification_report(y_test,pred)))