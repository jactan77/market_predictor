import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('encoded_100_users_data.csv')


X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Owns_MacBook', axis=1), data['Owns_MacBook'], random_state=0
)
knn = RandomForestClassifier()
knn.fit(X_train, y_train)
X_new = np.array([[65,0,0,1,0]])
predictions = knn.predict(X_new)
print('X_new shape: {}'.format(X_new.shape))
print('predictions: {}'.format(
    [predictions][0]
))
y_pred = knn.predict(X_test)
print("Score: {:.2f}".format(knn.score(X_test,y_test)))

