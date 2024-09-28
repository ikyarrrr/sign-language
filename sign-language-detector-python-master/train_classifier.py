import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Find the maximum length of the feature vectors
max_length = max(len(x) for x in data)

# Pad all feature vectors to the maximum length
padded_data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data])

# Convert labels to a NumPy array
labels = np.asarray(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model
model = RandomForestClassifier()

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Train best model
best_model = grid_search.best_estimator_

# Calibrate model
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(x_train, y_train)

# Predict and evaluate on test set
y_predict = calibrated_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'Test set accuracy: {score * 100:.2f}%')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': calibrated_model}, f)


