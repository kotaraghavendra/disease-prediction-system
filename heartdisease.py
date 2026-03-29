import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = np.genfromtxt('rdata.csv', delimiter=',', filling_values=0)

data = data[~np.isnan(data).any(axis=1)]

X = data[:, 0:12]
Y = data[:, 13]

Y = np.where(Y == 0, 0, 1)

print("Disease Prediction System")
print("Developed by Raghavendra")

pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', C=1)
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print("Accuracy:", accuracy)

x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], c=Y, cmap='coolwarm')
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Disease Prediction System using SVM")
plt.show()
