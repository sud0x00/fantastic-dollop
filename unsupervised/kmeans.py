# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Load the data

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-means model and fit it to the training data
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)

# Predict the cluster labels for the test set
y_pred = kmeans.predict(X_test)

# Evaluate the model by comparing the predicted labels to the true labels
score = kmeans.score(X_test, y_test)
