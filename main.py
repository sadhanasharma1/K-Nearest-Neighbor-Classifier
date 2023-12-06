# Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# Loading Dataset
iris = datasets.load_iris()

# Printing description
print(iris.DESCR)
# .data gives features
features = iris.data
# .target gives labels
labels = iris.target
print(features[0], labels[0])
# Training the classifier
# creating the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)
preds = clf.predict([[1, 1, 1, 1]])
print(preds)
