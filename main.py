import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset
data = pd.read_csv('fixed_user_preferences.csv')
print("Loading dataset {}".format(data.keys()))

# Preprocess the da

