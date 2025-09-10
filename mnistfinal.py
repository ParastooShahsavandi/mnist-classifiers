import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.special import gamma
from sklearn.base import BaseEstimator, ClassifierMixin

# Load and preprocess dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X / 255.0  # Normalize pixel values
X = np.array(X, dtype=float)  # Ensure all data is of type float64
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# SVM Classifiers
print("Running SVM Classifiers with Cross-Validation...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
linear_scores = cross_val_score(svm_linear, X_train, y_train, cv=kf)
print("Linear SVM Accuracy:", linear_scores.mean())

svm_poly = SVC(kernel='poly', degree=2)
svm_poly.fit(X_train, y_train)
poly_scores = cross_val_score(svm_poly, X_train, y_train, cv=kf)
print("Polynomial SVM (degree 2) Accuracy:", poly_scores.mean())

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
rbf_scores = cross_val_score(svm_rbf, X_train, y_train, cv=kf)
print("RBF SVM Accuracy:", rbf_scores.mean())

# Random Forest
print("Running Random Forest Classifier with Cross-Validation...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=kf)
print("Random Forest Accuracy:", rf_scores.mean())

# Beta Naive Bayes Classifier with Moments-Based Parameter Calculation
class BetaNaiveBayes(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Store the unique classes
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initializing parameters for each class
        self.alpha = np.zeros((n_classes, n_features))
        self.beta = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors[idx] = X_c.shape[0] / X.shape[0]
            
            # Moments-based approach for alpha and beta
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0) + 1e-9  # Add small constant to avoid division by zero
            K = (mean * (1 - mean) / var) - 1
            self.alpha[idx] = mean * K
            self.beta[idx] = (1 - mean) * K

    def predict(self, X):
        X = np.array(X, dtype=float)  # Ensure X is of type float64
        log_probs = []
        for idx, c in enumerate(self.classes_):
            log_prob = np.log(self.class_priors[idx])
            beta_pdf = (gamma(self.alpha[idx] + self.beta[idx]) /
                        (gamma(self.alpha[idx]) * gamma(self.beta[idx]))) * \
                        (X ** (self.alpha[idx] - 1)) * ((1 - X) ** (self.beta[idx] - 1))
            log_prob += np.sum(np.log(beta_pdf + 1e-9), axis=1)
            log_probs.append(log_prob)
        return self.classes_[np.argmax(log_probs, axis=0)]

print("Running Beta Naive Bayes Classifier with Cross-Validation...")
beta_nb = BetaNaiveBayes()
beta_nb.fit(X_train, y_train)
nb_scores = cross_val_score(beta_nb, X_train, y_train, cv=kf, scoring='accuracy')
print("Beta Naive Bayes Accuracy:", nb_scores.mean())

# Implement k-NN from Scratch
class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)  # Ensure X_train is a float array
        self.y_train = y
        self.classes_ = np.unique(y)

    def predict(self, X):
        X = np.array(X, dtype=float)  # Ensure X is a float array
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        x = np.array(x, dtype=float)  # Ensure x is float
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_neighbors = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_neighbors]
        return np.bincount(nearest_labels).argmax()

print("Running k-NN Classifier with Cross-Validation...")
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
knn_scores = cross_val_score(knn, X_train[:1000], y_train[:1000], cv=kf, scoring='accuracy')  # Limiting size for k-NN
print("k-Nearest Neighbors Accuracy:", knn_scores.mean())

# Timing the Classifiers
print("\nTiming the Classifiers...")
def time_model(model, X_train, y_train, X_test, y_test, name="Model"):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()
    
    print(f"{name} - Training Time: {end_train - start_train:.4f} seconds")
    print(f"{name} - Prediction Time: {end_test - start_test:.4f} seconds")
    print(f"{name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Timing each model
time_model(svm_linear, X_train, y_train, X_test, y_test, "Linear SVM")
time_model(svm_poly, X_train, y_train, X_test, y_test, "Polynomial SVM")
time_model(svm_rbf, X_train, y_train, X_test, y_test, "RBF SVM")
time_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
time_model(beta_nb, X_train, y_train, X_test, y_test, "Beta Naive Bayes")
time_model(knn, X_train[:1000], y_train[:1000], X_test[:1000], y_test[:1000], "k-NN")  # Limited size for k-NN timing
