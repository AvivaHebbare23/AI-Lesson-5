from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from colorama import Fore
import matplotlib.pyplot as plt
import time


mnist = fetch_openml('mnist_784', version=1)

X = mnist['data'] / 255.0  
y = mnist['target'].astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print(f"Test accuracy: {accuracy}") 

start = time.time()
for i in range(len(X_test)):
    model.predict(X_test.iloc[[i]])
end = time.time()

print(f"{Fore.YELLOW}It took the AI {Fore.GREEN}{round((end - start)/len(X_test), 5)}{Fore.YELLOW} seconds to predict each image.")

for i in range(5):  
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
    plt.show()
    time.sleep(0.3)