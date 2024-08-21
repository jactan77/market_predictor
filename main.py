import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('fixed_user_preferences.csv')
print("Loading dataset {}".format(data.keys()))


fig, ax = plt.subplots()



X = data.drop('UserID', axis = 1)
y = data['Owns_MacBook']

ax.scatter(X['Annual_Income'],y)
plt.show()