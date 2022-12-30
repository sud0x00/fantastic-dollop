from sklearn.svm import SVC

# Load the data
X = # input data
y = # output data

# Create the model
model = SVC()

# Fit the model to the data
model.fit(X, y)

# Make predictions on new data
predictions = model.predict(X_new)
