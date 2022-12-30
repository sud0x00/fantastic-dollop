from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Train the classifier using the training data
clf.fit(X_train, y_train)

# Test the classifier using the test data
accuracy = clf.score(X_test, y_test)

# Print the accuracy of the classifier
print('Accuracy:', accuracy)
