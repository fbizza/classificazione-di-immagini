from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = load_iris()
x = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

def normalize_data(x, y):  # such that σ = 1 and μ = 0
    sc = StandardScaler()
    scaler = sc.fit(x)
    x_scaled = scaler.transform(x)
    y_scaled = scaler.transform(y)
    return x_scaled, y_scaled


X_train, X_test = normalize_data(X_train, X_test)
clf = MLPClassifier((3, 3), tol=0.000001, activation="relu", max_iter=10000, solver="adam", n_iter_no_change=20, verbose=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predictions)))
