import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.compute_distance(x, x_train) for x_train in self.X_train]
            # Sort distances and get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get the most common label among the k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            prediction = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(prediction)
        return np.array(predictions)

    def compute_distance(self, X1, X2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X1 - X2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X1 - X2))
        else:
            raise ValueError("Unsupported distance metric")

# Data Preprocessing function
def preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Dropping unnecessary columns
    train_data = train_data.drop(['CustomerId', 'Surname'], axis=1)
    test_data = test_data.drop(['CustomerId', 'Surname'], axis=1)

    # One-hot encoding categorical variables
    train_data = pd.get_dummies(train_data, columns=['Geography', 'Gender'])
    test_data = pd.get_dummies(test_data, columns=['Geography', 'Gender'])

    # Ensure train and test have same columns
    X_train = train_data.drop('Exited', axis=1)
    y_train = train_data['Exited']
    X_test = test_data
    
    # Scaling numerical features (e.g., CreditScore, Age, Balance, etc.)
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.int64]:
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std

    return X_train.values, y_train.values, X_test.values

# Define cross-validation function
def cross_validate(X, y, knn, n_splits=5):
    fold_size = len(X) // n_splits
    scores = []
    for i in range(n_splits):
        # Create train/test split for cross-validation
        X_val = X[i*fold_size:(i+1)*fold_size]
        y_val = y[i*fold_size:(i+1)*fold_size]
        X_train = np.concatenate([X[:i*fold_size], X[(i+1)*fold_size:]])
        y_train = np.concatenate([y[:i*fold_size], y[(i+1)*fold_size:]])
        
        # Train and evaluate
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        scores.append(accuracy)
    
    return np.mean(scores)

# Load and preprocess data
X, y, X_test = preprocess_data('train.csv', 'test.csv')

# Create KNN model
knn = KNN(k=20, distance_metric='euclidean')

# Perform cross-validation
cv_score = cross_validate(X, y, knn)
print("Cross-validation accuracy:", cv_score)

# Train on full dataset and make predictions on test set
knn.fit(X, y)
test_predictions = knn.predict(X_test)

# Save test predictions
submission = pd.DataFrame({'id': pd.read_csv('test.csv')['id'], 'Exited': test_predictions})
submission.to_csv('submissions1.csv', index=False)
