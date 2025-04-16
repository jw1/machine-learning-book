import sys

from python_environment_check import check_packages
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, '..')

d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
    'pandas': '1.3.2'
}

check_packages(d)


# get the data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # only get petal length, width
y = iris.target # get the class names (as integer)

uniques = np.unique(y) # 50 each of 0, 1, 2.  filter to 1 of each
print(f'Class labels: {uniques}')

# 30% test data (45 examples), 70% training data (105 examples)... only 1 required
# random_state is a seed val (repeatable), and proportionally split up by class label (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1, stratify=y)

# standardize some features -- remove mean, scale to unit variance
# center the data around 0, divide by std dev so that spread is uniform
sc = StandardScaler()
sc.fit(X_train) # scaler estimates mean, std dev
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# training a perceptron with training data
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# testing the perceptron afterwards on the test set
y_pred = ppn.predict(X_test_std)

# count differences in different ways
print(f'Misclassified examples: { (y_test != y_pred).sum() }')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}%')

# score() combines predict() and accuracy_score()
print(f'Accuracy: {ppn.score(X_test_std, y_test):.3f}%')

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')  # circles, squares, triangles up, down, left
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # grab colors for each

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)  # ?, ?, ?, how transparent is the color, list of colors to use
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # if told which ones were test data, circle them by plotting them again as circles
    if test_idx:
        # X_test is all of the provided range
        # y_test is a horizontal array, so the
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')


# re-combine training, test data (these are the normalized ones)
X_combined_std = np.vstack((X_train_std, X_test_std)) # put test set at end of training set
y_combined = np.hstack((y_train, y_test)) # 1d array, so just append
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()