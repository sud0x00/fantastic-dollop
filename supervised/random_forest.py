from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data and split it into training and test sets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the random forests classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = clf.score(X_test, y_test)

print("Accuracy:", accuracy)
