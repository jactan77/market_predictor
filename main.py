import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

data = pd.read_csv('encoded_100_users_data.csv')


X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Owns_MacBook', axis=1), data['Owns_MacBook'], random_state=42
)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Score test: {:.2f}".format(tree.score(X_test,y_test)))
print("Score train: {:.2f}".format(tree.score(X_train, y_train)))

export_graphviz(tree, out_file='tree.dot', feature_names=data.drop('Owns_MacBook', axis=1).columns,
                class_names=['No', 'Yes'], rounded=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)