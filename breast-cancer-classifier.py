!pip install numpy
!pip install scikit-learn
!pip install seaborn

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

#Display variable names
print("Feature Names:", breast_cancer.feature_names)

#Display a data sample
print("\nSample Data:\n", breast_cancer.data[:1])

#Display a part of description
print("\nDescription:\n", breast_cancer.DESCR[:1063])

#Shape of data
print(dataset.data.shape)

#Target is a classification target aka correct classes
#x = data, y = correct classification
#Print a sample and length of target 
print(dataset.target[:100])
print(len(dataset.target))

#Split data into test and train with test constituting 20%
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=4, shuffle=True)

print(x_train.shape)

#Instantiate and train the model on the triaining dataset portion
NN = MLPClassifier()

NN.fit(x_train, y_train)

#Use classifier for prediction
y_pred = NN.predict(x_test)
print("Predicted classes:", y_pred)
print("Actual classes:", y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

#Display the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

#Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Purples', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#print the accuracy and the confusion matrix
print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix")
print(conf_mat)
