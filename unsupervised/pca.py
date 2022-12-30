from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load your data
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a PCA object with n_components=0.95
pca = PCA(n_components=0.95)

# Fit the PCA model on the training data
pca.fit(X_train)

# Transform the training and test data using the fitted PCA model
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
