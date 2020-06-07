import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import wget
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv'
#wget.download(url, '/Users/kevin/Desktop/drug200.csv')
df = pd.read_csv(r'C:/Users/kevin/Desktop/drug200.csv', delimiter = ',')
print(df.head(10))

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(["F", "M"])
X[:1] = le_sex.transform(X[:1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(["LOW", "NORMAL", "HIGH"])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(["NORMAL", "HIGH"])
X[:, 3] = le_Chol.transform(X[:, 3])

X[0:5]
y = df["Drug"]
y[0:5]


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
drugtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

#Check the accuracy of the model
print("DecisionTrees' Accuracy: ", metrics.accuracy_score(y_testset, predTree))


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
