from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)

X = mnist['data'] / 255.0 
y = mnist['target'].astype(int)

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, y_prediction)
print(f"Test accuracty: {accuracy}")

for i in range(5):
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {y_prediction[i]}, Actual: {Y_test.iloc[i]}")
    plt.show()