
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from sklearn import datasets, linear_model, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
inputFile = "text_data_r2.csv"
data = pd.read_csv(inputFile, header = 0)
data = pd.get_dummies(data)
#data = shuffle(data)
print(data)

#Add intercept
feature = ["intercept"] + list(data.columns)
#print( feature[37]+feature[372]+feature[10]+feature[77]+feature[17]+feature[86]+feature[19]+feature[394]+feature[39]+feature[10])
print("----")
print(data.columns)

#Extract Y column
Y = data[" y.1"]
question = data[" ?"]
feature = feature.remove(" y.1")
data = data.drop([' y.1', ' ok'], axis=1)


# pre-processing
Dmat = data.to_numpy(dtype=np.float64)
print(Dmat)

# add a column of 1s for the intercept term
Xmat = np.column_stack((np.ones(len(Dmat)), Dmat))


# extract outcome vector
n = len(Xmat)
text_X_train = Xmat[0:int(0.8*n)]
text_X_test = Xmat[int(0.8*n):, :]
text_y_train = Y[0:int(0.8*n)]
text_y_test = Y[int(0.8*n):]
base_predict_y = question[int(0.8*n):]

# Create linear regression object
regr = linear_model.LogisticRegression(C=1, penalty='l1', solver='liblinear')

# Train the model using the training sets
regr.fit(text_X_train, text_y_train)

# Make predictions using the testing set
text_y_pred = regr.predict(text_X_test)


# The coefficients
print("Coefficients: \n", regr.coef_)

print("\nUSING QUESTION MARKS")
print("Accuracy", round(accuracy_score(text_y_test, base_predict_y), 3))
print("F1 score", round(f1_score(text_y_test, base_predict_y, average='weighted'), 3))
print("Precision", round(precision_score(text_y_test, base_predict_y, average='weighted'), 3))
print("Recall", round(recall_score(text_y_test, base_predict_y, average='weighted', zero_division="warn"), 3))

print("\nUSING LOGISTIC REGRESSION")
print("Accuracy", round(accuracy_score(text_y_test, text_y_pred), 3))
print("F1 score", round(f1_score(text_y_test, text_y_pred), 3))
print("Precision", round(precision_score(text_y_test, text_y_pred), 3))
print("Recall", round(recall_score(text_y_test, text_y_pred), 3))

#########
#DECISION TREE

model = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
model.fit(text_X_train, text_y_train)
text_tree_y_pred = model.predict(text_X_test)


print("\nUSING TREE 1")
print("Accuracy", round(accuracy_score(text_y_test, text_tree_y_pred), 3))
print("F1 score", round(f1_score(text_y_test, text_tree_y_pred), 3))
print("Precision", round(precision_score(text_y_test, text_tree_y_pred), 3))
print("Recall", round(recall_score(text_y_test, text_tree_y_pred), 3))

model = DecisionTreeClassifier(max_depth=None, criterion="entropy", random_state=0)
model.fit(text_X_train, text_y_train)
text_tree_y_pred = model.predict(text_X_test)



print("\nUSING TREE MAX")
print("Accuracy", round(accuracy_score(text_y_test, text_tree_y_pred), 3))
print("F1 score", round(f1_score(text_y_test, text_tree_y_pred), 3))
print("Precision", round(precision_score(text_y_test, text_tree_y_pred), 3))
print("Recall", round(recall_score(text_y_test, text_tree_y_pred), 3))

model = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=0)
model.fit(text_X_train, text_y_train)
text_tree_y_pred = model.predict(text_X_test)



print("\nUSING TREE 4")
print("Accuracy", round(accuracy_score(text_y_test, text_tree_y_pred), 3))
print("F1 score", round(f1_score(text_y_test, text_tree_y_pred), 3))
print("Precision", round(precision_score(text_y_test, text_tree_y_pred), 3))
print("Recall", round(recall_score(text_y_test, text_tree_y_pred), 3))

figure = plt.figure(figsize=(25,30))
figure.set_size_inches(18.5, 10.5)
_ = tree.plot_tree(model, feature_names = feature, class_names = True, filled = True)

figure.savefig("decision_tree.pdf")

"""max = 0
iter = 0
for num in range(3, 500):
    model = DecisionTreeClassifier(max_depth=num, criterion="entropy", random_state=0)
    model.fit(text_X_train, text_y_train)
    text_tree_y_pred = model.predict(text_X_test)


    if round(accuracy_score(text_y_test, text_tree_y_pred), 3) > max:
        max = round(accuracy_score(text_y_test, text_tree_y_pred), 3)
        iter = num

print(str(iter) +  " : " + str(max))"""
##########
#NEURAL NET


