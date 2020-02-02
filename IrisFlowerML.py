#importing datasets from sklearn 
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
#training
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
#predicting
predictions=classifier.predict(x_test)
#checking for accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
