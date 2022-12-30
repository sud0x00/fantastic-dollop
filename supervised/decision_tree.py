from sklearn import tree

# Training data
X = [[0, 0], [1, 1]]
y = [0, 1]

# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier using the training data
clf = clf.fit(X, y)

# Make predictions on test data
X_test = [[2, 2], [3, 3]]
y_pred = clf.predict(X_test)
print(y_pred)  # Output: [1, 1]
