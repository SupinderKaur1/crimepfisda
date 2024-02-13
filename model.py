import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

# Load dataset (Replace 'your_dataset.csv' with the path to your actual CSV file)
df = pd.read_csv('CrimePM.csv')
print(df['AREA NAME'].unique())
# Create a LabelEncoder for each categorical feature
categorical_features = ['AREA NAME', 'Vict Sex', 'Day of week', 'Crm Cd Desc']
encoders = {col: LabelEncoder() for col in categorical_features}

# Fit and transform each categorical feature using the appropriate encoder
for col, encoder in encoders.items():
    df[col] = encoder.fit_transform(df[col])

# Feature selection (excluding non-numeric and non-predictive columns such as 'DR_NO')
predictive_features = ['AREA NAME', 'Vict Age', 'Vict Sex', 'Day of week']

X = df[predictive_features]
y = df['Crm Cd Desc']  # Target variable

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the Decision Tree classifier with max depth
model = DecisionTreeClassifier(max_depth=32, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Perform k-fold cross-validation
k = 10  # Number of folds
cv_scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
kf_accuracy = cv_scores.mean()
print(f'Average accuracy over {k} folds: {kf_accuracy}')

# y_test are the true labels and predictions are model predictions
conf_matrix = confusion_matrix(y_test, predictions)

# Calculating accuracy for each class
class_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

# Handling division by zero in case there are no instances of a class in y_test
class_accuracies = np.nan_to_num(class_accuracies)

# Calculating the average of the class-wise accuracies
average_class_wise_accuracy = np.mean(class_accuracies)

# Only calculate average over classes that have at least one instance in the test set
valid_class_indices = np.sum(conf_matrix, axis=1) > 0
valid_class_accuracies = class_accuracies[valid_class_indices]
average_valid_class_wise_accuracy = np.mean(valid_class_accuracies)

print(f'Average accuracy for classes present in the test set: {average_valid_class_wise_accuracy}')

print(f'Average class-wise accuracy: {average_class_wise_accuracy}')

# Save the model and encoders
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save each LabelEncoder
for col, encoder in encoders.items():
    with open(f'{col}_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)


